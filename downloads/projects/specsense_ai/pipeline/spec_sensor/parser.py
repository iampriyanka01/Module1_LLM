from typing import Dict, Any
from utils import get_logger, extract_json_from_text

logger = get_logger(__name__)

class OutputParser:
    """Parses LLM output into structured format for SpecSense AI."""

    def parse_llm_output(self, llm_output: Any) -> Dict[str, Any]:
        logger.info(f"Parsing LLM output of type {type(llm_output)}")

        # Case 1: None
        if llm_output is None:
            logger.error("LLM output is None.")
            return {
                "recommended_product": "Unknown",
                "meets": [],
                "partially_meets": [],
                "trade_offs": [],
                "review_snippets": [],
                "alternatives": []
            }

        # Case 2: Gemini structured response
        if hasattr(llm_output, "candidates"):
            try:
                parts = llm_output.candidates[0].content.parts
                text_parts = [p.text for p in parts if hasattr(p, "text") and p.text]
                llm_text = "\n".join(text_parts) if text_parts else ""
                logger.info(f"Extracted structured LLM output, text length: {len(llm_text)}")
            except Exception as e:
                logger.error(f"Failed to extract text from Gemini response: {e}")
                llm_text = ""
        else:
            # Case 3: Plain string
            llm_text = str(llm_output)
            logger.info(f"Parsing plain LLM output of length: {len(llm_text)}")

        # Try to extract JSON
        parsed_json = extract_json_from_text(llm_text)

        if parsed_json:
            logger.info("Successfully parsed LLM output to JSON.")
            return {
                "recommended_product": parsed_json.get("recommended_product", "Unknown"),
                "meets": parsed_json.get("meets", []),
                "partially_meets": parsed_json.get("partially_meets", []),
                "trade_offs": parsed_json.get("trade_offs", []),
                "review_snippets": parsed_json.get("review_snippets", []),
                "alternatives": parsed_json.get("alternatives", [])
            }
        else:
            logger.warning("No JSON found in LLM output.")
            return {
                "recommended_product": "Unknown",
                "meets": [],
                "partially_meets": [],
                "trade_offs": [],
                "review_snippets": [],
                "alternatives": []
            }
