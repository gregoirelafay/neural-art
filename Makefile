# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* neural-art/*.py

black:
	@black scripts/* neural-art/*.py

test:
	@coverage run --source=neuralart -m pytest -v --color=yes tests/*.py
	@coverage report -m --omit="*__init__.py,*legacy.py"
#"${VIRTUAL_ENV}/lib/python*"
# Add -s to display prints
ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr neural-art-*.dist-info
	@rm -fr neural-art.egg-info

# clean:
# 	@rm -f */version.txt
# 	@rm -f .coverage
# 	@rm -fr */__pycache__ __pycache__
# 	@rm -fr build dist *.dist-info *.egg-info
# 	@rm -fr */*.pyc

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=NeuralArtTeam
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

# ----------------------------------
#      GCP INFO
# ----------------------------------

PROJECT_ID=neural-art-323413

BUCKET_NAME=neural-art-bucket
BUCKET_FOLDER=data
BUCKET_FILE_NAME='wikiart'
BUCKET_TRAINING_FOLDER = 'trainings'

LOCAL_PATH_CSV='raw_data/wikiart/*.csv'
LOCAL_PATH_CHAN='raw_data/wikiart/csv_chan'

# LOCAL_PATH_IMAGE='raw_data/wikiart/dataset'
# LOCAL_PATH_IMAGE='raw_data/wikiart/wikiart-movement-genre_True-class_8-merge_mov-1-n_200_max'
# LOCAL_PATH_IMAGE='raw_data/wikiart/wikiart-movement-genre_True-class_8-merge_mov-1'
# LOCAL_PATH_IMAGE='raw_data/wikiart/wikiart-movement-genre_False-class_8-merge_mov-1'
LOCAL_PATH_IMAGE='raw_data/wikiart/wikiart-movement-genre_True-class_8-merge_mov-1-n_1440_max'

LOCAL_PATH_IMAGE_CHAN='raw_data/wikiart/dataset_chan'
# BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

VM_INSTANCE_NAME=neuralartvm
REGION=europe-west2

PYTHON_VERSION=3.7
FRAMEWORK=scikit-learn
RUNTIME_VERSION=1.15
PACKAGE_NAME=NeuralArt
FILENAME=trainer
JOB_NAME=neural_art_training_pipeline_$(shell date +'%Y%m%d_%H%M%S')

set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

upload_csv:
	@gsutil -m cp ${LOCAL_PATH_CSV} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}

upload_csv_chan:
	@gsutil -m cp -r ${LOCAL_PATH_CHAN} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}

upload_image:
	@gsutil -m cp -r ${LOCAL_PATH_IMAGE} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}

upload_image_chan:
	@gsutil -m cp -r ${LOCAL_PATH_IMAGE_CHAN} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}

run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

gcp_submit_training:
	@gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs \
		--scale-tier [TODO] \
    --master-machine-type n1-standard-16

gcp_ssh_connect:
	@gcloud compute ssh --project ${PROJECT_ID} --zone ${REGION} jupyter@${VM_INSTANCE_NAME} -- -L 8080:localhost:8080
