IMAGE = amr-registry-pre.caas.intel.com/tpi/ddpt:latest

all:
	cp ../conda-env.yml .
	docker build --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} --tag ${IMAGE} .
	docker push ${IMAGE}
