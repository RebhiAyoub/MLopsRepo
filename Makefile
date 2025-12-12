PYTHON = python
ENV_NAME = venv
REQUIREMENTS = requirements.txt
PREPARED_DATA = prepared_data.joblib
MODEL = random_forest_uber_model.joblib

.PHONY: all setup ci lint format security test data prepare train evaluate clean

all: setup ci prepare train evaluate
	@echo "✅ All pipeline steps completed successfully!"

# -----------------------------
# 1. ENV SETUP
# -----------------------------
setup:
	@echo "🔧 Creating virtual environment and installing dependencies..."
	@virtualenv $(ENV_NAME)
	@. $(ENV_NAME)/bin/activate && pip install --upgrade pip && pip install -r $(REQUIREMENTS)

# -----------------------------
# 2. CI / QUALITY
# -----------------------------
lint:
	@echo "🔍 Running pylint..."
	@$(ENV_NAME)/bin/pylint *.py || true

format:
	@echo "🎨 Formatting code using black..."
	@if [ -f $(ENV_NAME)/bin/black ]; then \
		$(ENV_NAME)/bin/black .; \
	else \
		echo "⚠️ black not installed. Installing..."; \
		$(ENV_NAME)/bin/pip install black; \
		$(ENV_NAME)/bin/black .; \
	fi

security:
	@echo "🛡️ Running bandit..."
	@$(ENV_NAME)/bin/bandit -r . || true

test:
	@echo "🧪 Running unit tests..."
	@$(ENV_NAME)/bin/pytest -v --disable-warnings

ci: lint format security test

# -----------------------------
# 3. DATA
# -----------------------------
data:
	@echo "📥 Loading data..."
	@$(ENV_NAME)/bin/python scripts/prepare_data.py --load

prepare:
	@echo "🔧 Preparing dataset..."
	@$(ENV_NAME)/bin/python scripts/prepare_data.py --prepare

# -----------------------------
# 4. TRAIN
# -----------------------------
train:
	@echo "🚀 Training model..."
	@$(ENV_NAME)/bin/python scripts/train_model.py

# -----------------------------
# 5. EVALUATE
# -----------------------------
evaluate:
	@echo "📊 Evaluating model..."
	@$(ENV_NAME)/bin/python scripts/evaluate_model.py

# -----------------------------
# 6. CLEAN
# -----------------------------
clean:
	@echo "🧹 Cleaning..."
	@rm -f $(PREPARED_DATA) $(MODEL)
	@echo "✨ Clean complete!"

# 7. DOCKER AUTOMATION

DOCKER_IMAGE_LOCAL = ayoubrebhi/ayoub_rebhi_4ds1_mlops
DOCKER_IMAGE_HUB   = ayoubrebhi/ayoubrebhi_mlops
DOCKER_TAG         = latest
CONTAINER_NAME     = uber_api_container

# Build Docker image
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t $(DOCKER_IMAGE_LOCAL):$(DOCKER_TAG) .

# Run container
docker-run:
	@echo "🚀 Running Docker container on port 8000..."
	docker run -d --name $(CONTAINER_NAME) -p 8000:8000 $(DOCKER_IMAGE_LOCAL):$(DOCKER_TAG)

# Stop container
docker-stop:
	@echo "🛑 Stopping Docker container..."
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)

# Tag image for Docker Hub
docker-tag:
	@echo "🏷️ Tagging image for Docker Hub..."
	docker tag $(DOCKER_IMAGE_LOCAL):$(DOCKER_TAG) $(DOCKER_IMAGE_HUB):$(DOCKER_TAG)

# Push to Docker Hub
docker-push: docker-tag
	@echo "📤 Pushing image to Docker Hub..."
	docker push $(DOCKER_IMAGE_HUB):$(DOCKER_TAG)

# Full deployment pipeline: build → tag → push → run
docker-deploy: docker-build docker-push docker-run
	@echo "🚀 Deployment complete!"
