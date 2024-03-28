include ../config.mk

build: $(addprefix build-,$(runs))
clean: $(addprefix clean-,$(runs))

build-%:
	$(MAKE) -C $* build

clean-%:
	$(MAKE) -C $* clean