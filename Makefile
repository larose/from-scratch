.PHONY: check
check:
	cd linear-regression && make check

.PHONY: install
install:
	cd linear-regression && make install

.PHONY: test
test:
	cd linear-regression && make test
