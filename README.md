# pattern_synthesizer_api
Steps to generate patterns in test mode
1. cd into root directory 
2. uvicorn main:app --reload --port 8080
3. open postman
..* 127.0.0.1:8080/dataset #To load dataset
..* 127.0.0.1:8080/test/{# of iteration}/{# of annotation per iteration}
....* results/test_results.json should hold new results, use results/viz_result.ipynb to visualize results
