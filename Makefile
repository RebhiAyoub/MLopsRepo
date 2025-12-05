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
