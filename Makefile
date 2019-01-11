ASSIGNMENT_NUMBER = 1 2 3 4 5 6

output:
	jupyter nbconvert --to markdown Analysis.ipynb
	jupyter nbconvert --to pdf Analysis.ipynb

test:
	@echo "RUNNING UNIT TESTS..."
	@echo ""
	@for i in $(ASSIGNMENT_NUMBER) ; do \
		echo "=== RUNNING UNIT TEST FOR ASSIGNMENT $$i ==="; \
		python3 assignment$$i/unit_tests.py -b; \
	done

coverage:
	@echo "RUNNING COVERAGE TEST..."
	@echo ""
	@coverage erase;
	@for i in $(ASSIGNMENT_NUMBER) ; do \
		echo "=== RUNNING UNIT TEST FOR ASSIGNMENT $$i ==="; \
		coverage run -a --source code assignment$$i/unit_tests.py -b; \
	done
	@echo "";
	@echo "=== Final Report ===";
	@coverage report;

clean:
	@coverage erase;
	@for i in $(ASSIGNMENT_NUMBER) ; do \
		rm -f assignment$$i/.coverage; \
		rm -rf assignment$$i/htmlcov; \
	done
	@rm -f Analysis.md
	@rm -f Analysis.pdf

