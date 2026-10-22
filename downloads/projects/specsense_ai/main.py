from pathlib import Path
import sys
from pipeline.spec_sensor.sensor import SpecSensor
from utils import get_logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

logger = get_logger(__name__)

def main():
    """Test the scam detection system with dynamic message."""
    detector = SpecSensor()
   
    test_msg = str(input("Enter your Message: "))
   
    try:
        logger.info("Running scam detection")

        #Excecution will start from here and will go through the 3-step pipeline: build_prompt -> execute -> parse_llm_output
        result = detector.detect(test_msg)
       
        print(f"Test Message: {test_msg}")
        print(f"Detection Result: {result}")
       
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()