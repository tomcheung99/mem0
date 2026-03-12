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
            try:
                return _orig(self, messages, response_format=response_format,
                             tools=tools, tool_choice=tool_choice, **kwargs)
            except Exception as e:
                err = str(e).lower()
                if "response_format" in err or ("invalid" in err and "input" in err):
                    logging.debug(
                        "mem0 patch: response_format rejected by endpoint, retrying without it: %s", e
                    )
                    return _orig(self, messages, response_format=None,
                                 tools=tools, tool_choice=tool_choice, **kwargs)
                raise

        _mod.OpenAILLM.generate_response = _patched
        logging.info("mem0 patch: OpenAI response_format fallback applied")
    except Exception as e:
        logging.warning("mem0 patch: could not apply patch (non-fatal): %s", e)


apply()
