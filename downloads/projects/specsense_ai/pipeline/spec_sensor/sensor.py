from typing import List, Dict, Any
from .builder import build_prompt
from .executor import LLMExecutor
from .parser import OutputParser
from utils import get_logger

logger = get_logger(__name__)

class SpecSensor:
    """Orchestrates the spec sensor pipeline."""
    
    def __init__(self, strategy: str = "react") -> None:
        """Initializes the pipeline components."""
        self.executor = LLMExecutor()
        self.parser = OutputParser()
        self.strategy = strategy
        logger.info(f"Initialized spec sensor with strategy: {self.strategy}")

    def detect(self, message: str) -> Dict[str, Any]:
        """Runs spec sensor on a single query: build prompt -> execute -> parse."""
        logger.info(f"Starting spec sensor for message length: {len(message)}")
        try:
            # The 3-step pipeline
            prompt = build_prompt(message, self.strategy)
            raw_response = self.executor.execute(prompt)
            parsed_result = self.parser.parse_llm_output(raw_response)

            logger.info(f"Spec sensor run is successful. Recommended: {parsed_result.get('recommended_product', 'Unknown')}")
            return parsed_result

        except Exception as e:
            logger.error(f"spec sensor pipeline failed: {e}")
            # Re-raise the exception to be handled by the caller (UI layer)
            raise

    def detect_batch(self, messages: List[str]) -> List[Dict[str, Any]]:
        """Runs spec sensor on a list of messages."""
        total_messages = len(messages)
        logger.info(f"Starting batch spec sensor for {total_messages} messages.")
       
        results: List[Dict[str, Any]] = []
       
        for i, message in enumerate(messages):
            try:
                result = self.detect(message)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process message {i+1}/{total_messages}: {e}")
                # Append a fallback error result for the failed message
                error_result = {
                    "recommended_product": "Unknown",
                    "meets": [],
                    "partially_meets": [],
                    "trade_offs": [f"processing_error: {e}"],
                    "review_snippets": [],
                    "alternatives": []
                }
                results.append(error_result)
       
        successful = sum(1 for r in results if r.get("recommended_product") != "Unknown")
        logger.info(f"Batch processing complete. {successful}/{total_messages} succeeded.")
        return results