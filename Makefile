install:
	pip install -r requirements.txt
	
test:
	pytest --verbose ./test

format:
	black *.py

lint:
	pylint --disable=R,C *.py

.PHONY: test
