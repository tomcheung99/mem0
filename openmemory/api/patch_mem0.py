"""
Monkey-patch mem0's OpenAI LLM so it retries without response_format when the
endpoint rejects it (e.g. Vercel AI Gateway returns HTTP 400 for that param).

Import this module once at application startup (before any mem0 usage).
"""
import logging


def apply():
    try:
        import mem0.llms.openai as _mod

        _orig = _mod.OpenAILLM.generate_response

        def _patched(self, messages, response_format=None, tools=None, tool_choice="auto", **kwargs):
            # When response_format is used the endpoint may reject it and fall back to
            # plain-text JSON generation.  Without structured-output enforcement, the
            # model can produce longer payloads that get truncated at the default
            # max_tokens=2000 limit, resulting in unterminated JSON strings.  Raise the
            # floor to 4000 for any JSON call so the retry also benefits from the larger
            # budget (params is reused inside _orig's own retry logic).
            if response_format is not None and "max_tokens" not in kwargs:
                if self.config.max_tokens < 4000:
                    kwargs["max_tokens"] = 4000
            return _orig(self, messages, response_format=response_format,
                         tools=tools, tool_choice=tool_choice, **kwargs)

        _mod.OpenAILLM.generate_response = _patched
        logging.info("mem0 patch: OpenAI response_format fallback applied")
    except Exception as e:
        logging.warning("mem0 patch: could not apply patch (non-fatal): %s", e)


apply()
