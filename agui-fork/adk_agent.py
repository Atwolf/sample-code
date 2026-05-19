        When using ADK's ResumabilityConfig(is_resumable=True), the Runner
        automatically persists FunctionCall events before pausing. This allows
        us to let ADK handle the pause/resume flow naturally instead of
        returning early at LRO tool calls.

        Returns:
            True if using from_app() with ResumabilityConfig.is_resumable=True
        """
        if self._app is None:
            return False
        resumability_config = getattr(self._app, 'resumability_config', None)
        if resumability_config is None:
            return False
        return getattr(resumability_config, 'is_resumable', False)

    def _root_agent_needs_invocation_id(self) -> bool:
        """Check if the agent topology requires invocation_id for HITL resumption.

        Composite orchestrators (SequentialAgent, LoopAgent) store internal
        state (e.g. current_sub_agent position) that can only be restored via
        populate_invocation_agent_states(), which requires invocation_id.

        This returns True when:
        - The root agent itself is a composite orchestrator, OR
        - Any agent in the sub-agent tree is a composite orchestrator
          (e.g. LlmAgent → LlmAgent → SequentialAgent).

        Standalone LlmAgents (including those with only LlmAgent transfer
        targets) do NOT need invocation_id. Passing it triggers
        _get_subagent_to_resume() which raises ValueError.

        Returns:
            True if the topology contains a composite orchestrator
        """
        from google.adk.agents import LoopAgent, SequentialAgent
        composite_types = (SequentialAgent, LoopAgent)

        root = self._adk_agent
        if root is None and self._app is not None:
            root = getattr(self._app, 'root_agent', None)
        if root is None:
            return False
        if isinstance(root, composite_types):
            return True

        def _has_composite_descendant(agent):
            for sub in getattr(agent, 'sub_agents', None) or []:
                if isinstance(sub, composite_types):
                    return True
                if _has_composite_descendant(sub):
                    return True
            return False

        return _has_composite_descendant(root)

    @staticmethod
    def _find_function_call_invocation_id(session, tool_call_id: str) -> Optional[str]:
        """Find the invocation_id of the event that authored a FunctionCall.

        ADK 1.30+ derives the effective invocation_id for tool-result submissions
        by looking up the matching FunctionCall event in session history. We read
        the same attribute here so that any FunctionResponse we pre-append carries
        a consistent invocation_id with the upstream FunctionCall.

        Returns None if no matching FunctionCall event is found.
        """
        events = getattr(session, "events", None) or []
        for event in events:
            content = getattr(event, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if not parts:
                continue
            for part in parts:
                fc = getattr(part, "function_call", None)
                fc_id = getattr(fc, "id", None) if fc else None
                if fc_id and fc_id == tool_call_id:
                    return getattr(event, "invocation_id", None)
        return None

    @classmethod
    def from_app(
        cls,
        app: "App",
        # User identification (still needed - not in App)
        user_id: Optional[str] = None,
        user_id_extractor: Optional[Callable[[RunAgentInput], str]] = None,
        # ADK Services (App does NOT contain these - still passed to Runner separately)
        session_service: Optional[BaseSessionService] = None,
        session_manager: Optional[SessionManager] = None,
        artifact_service: Optional[BaseArtifactService] = None,
        memory_service: Optional[BaseMemoryService] = None,
        credential_service: Optional[BaseCredentialService] = None,
        # Configuration
        run_config_factory: Optional[Callable[[RunAgentInput], ADKRunConfig]] = None,
        use_in_memory_services: bool = True,
        plugin_close_timeout: float = 5.0,
        # Execution limits
        execution_timeout_seconds: int = 600,
        tool_timeout_seconds: int = 300,
        max_concurrent_executions: int = 10,
        # Session management
        session_timeout_seconds: Optional[int] = 1200,
        cleanup_interval_seconds: int = 300,
        max_sessions_per_user: Optional[int] = None,    # No limit by default
        delete_session_on_cleanup: bool = True,
        save_session_to_memory_on_cleanup: bool = True,
        # AG-UI specific
        predict_state: Optional[Iterable[PredictStateMapping]] = None,
        emit_messages_snapshot: bool = False,
        streaming_function_call_arguments: bool = False,
        # Session identity
        use_thread_id_as_session_id: bool = False,
        # Agent capabilities
        capabilities: Optional[Dict[str, Any]] = None,
        sub_agents: Optional[Union[List["ADKAgent"], Dict[str, "ADKAgent"]]] = None,
        select_agent_from_state: Optional[Callable[[RunAgentInput, Any, Dict[str, "ADKAgent"]], Any]] = None,
    ) -> "ADKAgent":
        """Create ADKAgent from an ADK App instance.

        This is the recommended way to create an ADKAgent when you want access to
        App-level features like resumability, context caching, and plugins.

        The App object bundles together the root agent, plugins, and configuration
        that would otherwise need to be passed separately. Using from_app() enables:
        - Plugin support (logging, tracing, custom plugins)
        - Resumability configuration for pause/resume workflows
        - Context caching configuration for LLM optimization
        - Events compaction configuration

        Args:
            app: The ADK App instance containing the root agent and configuration
            user_id: Static user ID for all requests
            user_id_extractor: Function to extract user ID dynamically from input
            session_service: Session management service (defaults to InMemorySessionService).
                See ADKAgent.__init__ for details.
            session_manager: Pre-constructed SessionManager to use. When provided,
                ``session_service`` and the session-cleanup configuration arguments
                are ignored. See ADKAgent.__init__ for details.
            artifact_service: File/artifact storage service
            memory_service: Conversation memory and search service
            credential_service: Authentication credential storage
            run_config_factory: Function to create RunConfig per request
            use_in_memory_services: Use in-memory implementations for unspecified services
            plugin_close_timeout: Timeout for plugin close methods (requires ADK 1.19+)
            execution_timeout_seconds: Timeout for entire execution
            tool_timeout_seconds: Timeout for individual tool calls
            max_concurrent_executions: Maximum concurrent background executions
            session_timeout_seconds: Session timeout in seconds
            cleanup_interval_seconds: Interval for session cleanup
            predict_state: Configuration for predictive state updates
            emit_messages_snapshot: Whether to emit MessagesSnapshotEvent at end of runs
            streaming_function_call_arguments: Whether to enable streaming of function
                call arguments from Gemini 3+ models. Requires google-adk >= 1.24.0.
            use_thread_id_as_session_id: When True, use the AG-UI thread_id directly
                as the ADK session_id. See ADKAgent.__init__ for details.
            capabilities: Optional dictionary of agent capabilities conforming to
                the AG-UI AgentCapabilities schema. See ADKAgent.__init__ for details.
            sub_agents: Optional selectable AG-UI ADKAgent registry. See
                add_adk_fastapi_endpoint for details.
            select_agent_from_state: Optional request-scoped selector. See
                add_adk_fastapi_endpoint for details.

        Returns:
            ADKAgent instance configured to use the App

        Example:
            from google.adk.apps import App
            from google.adk.agents import Agent

            app = App(
                name="my_assistant",
                root_agent=Agent(name="assistant", model="gemini-2.5-flash", ...),
                plugins=[LoggingPlugin()],
            )
            agent = ADKAgent.from_app(app, user_id="demo_user")
        """
        # Import App at runtime to avoid circular imports
        from google.adk.apps import App as AppClass

        if not isinstance(app, AppClass):
            raise TypeError(f"Expected App instance, got {type(app).__name__}")

        instance = cls(
            adk_agent=app.root_agent,
            app_name=app.name,
            user_id=user_id,
            user_id_extractor=user_id_extractor,
            session_service=session_service,
            session_manager=session_manager,
            artifact_service=artifact_service,
            memory_service=memory_service,
            credential_service=credential_service,
            run_config_factory=run_config_factory,
            use_in_memory_services=use_in_memory_services,
            execution_timeout_seconds=execution_timeout_seconds,
            tool_timeout_seconds=tool_timeout_seconds,
            max_concurrent_executions=max_concurrent_executions,
            session_timeout_seconds=session_timeout_seconds,
            cleanup_interval_seconds=cleanup_interval_seconds,
            max_sessions_per_user=max_sessions_per_user,
            delete_session_on_cleanup=delete_session_on_cleanup,
            save_session_to_memory_on_cleanup=save_session_to_memory_on_cleanup,
            predict_state=predict_state,
            emit_messages_snapshot=emit_messages_snapshot,
            streaming_function_call_arguments=streaming_function_call_arguments,
            use_thread_id_as_session_id=use_thread_id_as_session_id,
            capabilities=capabilities,
            sub_agents=sub_agents,
            select_agent_from_state=select_agent_from_state,
        )
        # Store App for per-request App creation with modified agents
        instance._app = app
        instance._plugin_close_timeout = plugin_close_timeout
        return instance

    def get_capabilities(self) -> Optional[Dict[str, Any]]:
        """Return a copy of the agent's declared capabilities, or None if not configured.

        These capabilities conform to the AG-UI AgentCapabilities schema and are
        served by the GET /capabilities endpoint when using add_adk_fastapi_endpoint().
        """
        if self._capabilities is None:
            return None
        return copy.deepcopy(self._capabilities)

    def _get_session_metadata(self, thread_id: str, user_id: str) -> Optional[Tuple[str, str, str]]:
        """Get session metadata for a (thread_id, user_id) pair efficiently.

        Args:
            thread_id: The AG-UI thread_id to lookup
            user_id: The user identifier to scope the lookup (use "" only when explicitly anonymous)

        Returns:
            Tuple of (session_id, app_name, user_id) or None if not found
        """
        return self._session_lookup_cache.get((thread_id, user_id))

    def _get_backend_session_id(self, thread_id: str, user_id: str) -> Optional[str]:
        """Get the backend session_id for a (thread_id, user_id) pair.

        Args:
            thread_id: The AG-UI thread_id to lookup
            user_id: The user identifier to scope the lookup (use "" only when explicitly anonymous)

        Returns:
            The backend session_id or None if not found
        """
        metadata = self._session_lookup_cache.get((thread_id, user_id))
        return metadata[0] if metadata else None
    
