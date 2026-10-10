RUN = uv run python -m decision_rerank


.PHONY: all resume score setup fetch smoke benchmark cascade calibration stability longcontext prompts layout validity analyses figures tables verify clean

all: setup fetch score prompts layout validity stability benchmark calibration cascade longcontext figures tables verify

resume:
	$(MAKE) all

score:
	$(RUN).benchmark --stage score

setup:
	uv sync

fetch:
	$(RUN).fetch all

smoke:
	$(RUN).smoke

benchmark:
	$(RUN).benchmark

cascade:
	$(RUN).cascade

calibration:
	$(RUN).calibration

stability:
	$(RUN).stability

longcontext:
	$(RUN).longcontext

prompts:
	$(RUN).prompts

layout:
	$(RUN).layout

validity:
	$(RUN).validity

analyses: cascade calibration stability longcontext prompts layout validity

figures:
	$(RUN).figures

tables:
	$(RUN).tables

verify:
	uv run pytest -q

clean:
	rm -rf results plots tables .cache logs
