include ../../config.mk

build: $(patsubst %,bin/%.exe,$(cases))

bin/%.exe: $(cases_dir)/%.fir compile.sh
	@mkdir -p bin build build/$*
	@cp $< build/$*/$*.fir
	./compile.sh build/$* $*.fir $(shell pwd)/$@

clean:
	rm -rf build bin