# GitHub Issues for Sentimental Analysis in Healthcare Project

## Issue 1: Missing scikit-learn Dependency in requirements.txt

**Title:** Add scikit-learn to requirements.txt

**Body:**
```markdown
## Problem
The project uses scikit-learn extensively (in `Model.py`, `evaluation.py`, `TF_IDF.py`, and `main.py`), but `scikit-learn` is not listed in `requirements.txt`. This will cause installation failures for new users.

## Expected Behavior
Running `pip install -r requirements.txt` should install all required dependencies, including scikit-learn.

## Current Behavior
Users need to manually install scikit-learn after installing from requirements.txt, or the code will fail with `ModuleNotFoundError: No module named 'sklearn'`.

## Solution
Add `scikit-learn` to `requirements.txt` with an appropriate version constraint (e.g., `scikit-learn>=1.0.0`).

## Files Affected
- `requirements.txt`
```

---

## Issue 2: Code Duplication - Functions Redefined in main.py

**Title:** Remove code duplication - functions already exist in separate modules

**Body:**
```markdown
## Problem
`main.py` redefines several functions that already exist in separate module files:
- `load_and_preprocess_data()` exists in both `preprocessing.py` and `main.py`
- `apply_sentiment_labeling()` exists in both `sentiment_labeling.py` and `main.py`
- `apply_tfidf()` exists in both `TF_IDF.py` and `main.py`
- `train_model()` exists in both `Model.py` and `main.py`
- `evaluate_model()` exists in both `evaluation.py` and `main.py`

This creates maintenance issues and confusion about which implementation is actually being used.

## Expected Behavior
`main.py` should import and use functions from their respective modules without redefining them.

## Current Behavior
Functions are defined inline in `main.py`, making the module imports at the top of the file unused or conflicting.

## Solution
1. Remove the duplicate function definitions from `main.py`
2. Ensure all imports are properly used
3. Keep only the orchestration logic in `main.py`

## Files Affected
- `main.py`
```

---

## Issue 3: Inconsistent Model Implementation

**Title:** Model.py and main.py use different classifiers

**Body:**
```markdown
## Problem
There's an inconsistency in model implementation:
- `Model.py` uses `LogisticRegression` with `test_size=0.2`
- `main.py` uses `RandomForestClassifier` with `test_size=0.3`

This creates confusion about which model is actually being used when importing from `Model.py`.

## Expected Behavior
The model implementation should be consistent. Either:
1. `main.py` should use the model from `Model.py`, OR
2. `Model.py` should be updated to match what's used in `main.py`, OR
3. The model choice should be configurable

## Current Behavior
If someone imports `train_model` from `Model.py`, they get a different model than what's used in `main.py`.

## Solution
1. Decide on a single model implementation
2. Update `main.py` to use the function from `Model.py` (or vice versa)
3. Consider making the model type configurable via parameters

## Files Affected
- `Model.py`
- `main.py`
```

---

## Issue 4: Evaluation Function Assumes Binary Classification

**Title:** Fix evaluation.py to handle multi-class classification correctly

**Body:**
```markdown
## Problem
The `evaluate_model()` function in `evaluation.py` assumes binary classification:
- Line 115: `y_proba = model.predict_proba(X_test)[:, 1]` - hardcodes index 1 for positive class
- ROC and Precision-Recall curves are calculated for binary classification only

However, the sentiment classification has 3 classes: Negative (-1), Neutral (0), and Positive (1).

## Expected Behavior
The evaluation should properly handle multi-class classification:
- Use appropriate metrics for multi-class (e.g., macro/micro averages)
- Generate ROC curves for each class (one-vs-rest) or use multi-class ROC
- Update confusion matrix labels to match actual class labels

## Current Behavior
The evaluation may work but is not correctly configured for the 3-class problem, potentially giving misleading results.

## Solution
1. Update `evaluate_model()` to handle multi-class classification
2. Use `roc_auc_score` with `multi_class` parameter
3. Generate per-class ROC curves or use macro/micro averaging
4. Ensure confusion matrix labels match the actual class values (-1, 0, 1)

## Files Affected
- `evaluation.py`
```

---

## Issue 5: Test Dataset Not Used Separately

**Title:** Test dataset should be used for final evaluation, not concatenated with training data

**Body:**
```markdown
## Problem
The current implementation concatenates train and test datasets in `preprocessing.py`:
```python
data = pd.concat([trainDataset, testDataset])
```

This means the test set is used during training, which violates the principle of keeping test data separate for unbiased evaluation.

## Expected Behavior
1. Training data should be used for training and validation
2. Test data should be kept separate and only used for final evaluation
3. The test set should not be seen during model training or hyperparameter tuning

## Current Behavior
Both train and test datasets are combined, and then a random split is performed, meaning the original test set boundaries are lost.

## Solution
1. Keep train and test datasets separate
2. Use train dataset for training/validation split
3. Use test dataset only for final evaluation
4. Update `preprocessing.py` to return separate train and test dataframes
5. Update `main.py` to handle separate train/test datasets

## Files Affected
- `preprocessing.py`
- `main.py`
```

---

## Issue 6: README References Non-existent File

**Title:** Fix README - mentions sentiment.py but actual file is main.py

**Body:**
```markdown
## Problem
The README.md file (line 64) instructs users to run:
```bash
python sentiment.py
```

However, the actual entry point file is `main.py`, not `sentiment.py`.

## Expected Behavior
The README should provide correct instructions that match the actual project structure.

## Current Behavior
Users following the README will get a `FileNotFoundError` when trying to run `sentiment.py`.

## Solution
Update the README to reference `main.py` instead of `sentiment.py`.

## Files Affected
- `README.md`
```

---

## Issue 7: Large Blocks of Commented Code Should Be Removed

**Title:** Clean up commented-out code blocks

**Body:**
```markdown
## Problem
Multiple files contain large blocks of commented-out code:
- `main.py` has ~75 lines of commented code at the top
- `preprocessing.py` has ~40 lines of commented code
- `sentiment_labeling.py` has commented NLTK download calls

This makes the codebase harder to read and maintain.

## Expected Behavior
Code should be clean and free of unnecessary commented blocks. If code is needed for reference, it should be in version control history, not in the source files.

## Current Behavior
Large commented blocks clutter the files and make it difficult to understand the current implementation.

## Solution
1. Remove all commented-out code blocks
2. If any commented code contains important logic, either:
   - Uncomment and use it if needed, OR
   - Remove it (it's preserved in git history)

## Files Affected
- `main.py`
- `preprocessing.py`
- `sentiment_labeling.py`
- `TF_IDF.py` (if any)
```

---

## Issue 8: Missing Error Handling

**Title:** Add error handling for file operations and data processing

**Body:**
```markdown
## Problem
The code lacks error handling for:
- File reading operations (what if files don't exist?)
- Data processing steps (what if data is malformed?)
- Model training (what if training fails?)
- NLTK resource downloads (what if download fails?)

## Expected Behavior
The application should:
- Handle missing files gracefully with clear error messages
- Validate data before processing
- Provide informative error messages to users
- Handle edge cases (empty datasets, missing columns, etc.)

## Current Behavior
If any step fails, the entire program crashes with a cryptic error message.

## Solution
1. Add try-except blocks around file I/O operations
2. Add data validation checks
3. Add meaningful error messages
4. Consider using logging instead of print statements for better error tracking

## Files Affected
- `main.py`
- `preprocessing.py`
- `Model.py`
- `evaluation.py`
```

---

## Issue 9: Hardcoded File Paths

**Title:** Make file paths configurable via command-line arguments or config file

**Body:**
```markdown
## Problem
File paths are hardcoded in `main.py`:
```python
train_file = "drugsComTrain_raw.tsv"
test_file = "drugsComTest_raw.tsv"
```

This makes it difficult to:
- Use different datasets
- Run the code with files in different locations
- Test with different data files

## Expected Behavior
The application should accept file paths as:
- Command-line arguments, OR
- Configuration file, OR
- Environment variables

## Current Behavior
Users must modify the source code to use different files.

## Solution
1. Add command-line argument parsing (using `argparse`)
2. Allow users to specify train and test file paths
3. Provide default values for backward compatibility
4. Optionally support a config file for more complex setups

## Files Affected
- `main.py`
```

---

## Issue 10: Inconsistent NLTK Download Handling

**Title:** Standardize NLTK resource download handling

**Body:**
```markdown
## Problem
NLTK resource downloads are handled inconsistently:
- `preprocessing.py` has active `nltk.download()` calls (lines 50-52)
- `main.py` has commented-out `nltk.download()` calls (lines 92-94)
- `sentiment_labeling.py` has commented-out download call (line 5)

This can cause issues:
- Downloads happen automatically in some cases but not others
- Users may encounter errors if resources aren't downloaded
- No check if resources already exist before downloading

## Expected Behavior
NLTK resources should be:
- Checked for existence before attempting download
- Downloaded only if missing
- Handled consistently across all files
- Documented in README

## Current Behavior
Inconsistent download behavior may cause runtime errors or unnecessary downloads.

## Solution
1. Create a utility function to safely download NLTK resources
2. Check if resources exist before downloading
3. Use this utility consistently across all files
4. Document required NLTK resources in README

## Files Affected
- `preprocessing.py`
- `main.py`
- `sentiment_labeling.py`
- Consider creating a new `utils.py` for shared utilities
```

---

## Issue 11: Missing Type Hints

**Title:** Add type hints to improve code readability and maintainability

**Body:**
```markdown
## Problem
The codebase lacks type hints, making it:
- Harder to understand function signatures
- Difficult for IDEs to provide autocomplete
- More prone to type-related bugs
- Less maintainable

## Expected Behavior
All functions should have type hints for:
- Parameters
- Return values
- Complex data structures

## Current Behavior
Functions have no type information, requiring readers to infer types from usage.

## Solution
Add type hints using Python's `typing` module:
- Import necessary types (`List`, `Dict`, `Tuple`, `Optional`, etc.)
- Add type hints to all function signatures
- Use type hints for complex variables where helpful

## Files Affected
- All Python files in the project
```

---

## Issue 12: Replace Print Statements with Logging

**Title:** Implement proper logging instead of print statements

**Body:**
```markdown
## Problem
The code uses `print()` statements for output:
- No log levels (info, warning, error, debug)
- Cannot easily disable or redirect output
- No timestamps or context information
- Difficult to debug production issues

## Expected Behavior
The application should use Python's `logging` module:
- Different log levels for different types of messages
- Configurable output (console, file, both)
- Timestamps and context information
- Easy to enable/disable debug output

## Current Behavior
All output goes to stdout via print statements, making it difficult to control verbosity or save logs.

## Solution
1. Replace print statements with appropriate logging calls
2. Configure logging with proper levels and format
3. Add timestamps and context to log messages
4. Allow log level configuration via command-line or config

## Files Affected
- `main.py`
- `evaluation.py`
- Potentially other files with print statements
```

---

## Issue 13: Missing Data Validation

**Title:** Add data validation and sanity checks

**Body:**
```markdown
## Problem
The code doesn't validate:
- Data file existence before reading
- Required columns in datasets
- Data types of columns
- Empty datasets after preprocessing
- Missing values handling strategy
- Class distribution (potential class imbalance)

## Expected Behavior
The application should:
- Validate file existence and format
- Check for required columns
- Verify data types
- Handle edge cases (empty data, all missing values, etc.)
- Warn about class imbalance if present

## Current Behavior
The code may fail with cryptic errors if data doesn't meet expectations, or silently produce incorrect results.

## Solution
1. Add validation functions to check data integrity
2. Verify required columns exist
3. Check data types match expectations
4. Validate data after each preprocessing step
5. Add warnings for potential issues (class imbalance, missing data, etc.)

## Files Affected
- `preprocessing.py`
- `main.py`
- Consider creating a `validation.py` module
```

---

## Issue 14: Inconsistent Sentiment Labeling Logic

**Title:** Sentiment labeling uses different methods in different files

**Body:**
```markdown
## Problem
Sentiment labeling is implemented differently:
- `sentiment_labeling.py` uses VADER sentiment analyzer to compute compound scores
- `main.py` (line 131) uses a simple rating-based approach: `lambda x: 1 if x > 3 else -1 if x < 3 else 0`

This creates confusion about which method is actually being used.

## Expected Behavior
There should be a single, consistent method for sentiment labeling, or clear documentation about when to use each method.

## Current Behavior
The `main.py` implementation overrides the VADER-based approach from `sentiment_labeling.py`, so the VADER analyzer is never actually used despite being imported.

## Solution
1. Decide on a single sentiment labeling approach
2. If using VADER, ensure it's actually used (remove the rating-based approach from main.py)
3. If using rating-based, remove VADER dependencies
4. Document the chosen approach in README

## Files Affected
- `main.py`
- `sentiment_labeling.py`
```

---

## Issue 15: Missing Unit Tests

**Title:** Add unit tests for core functionality

**Body:**
```markdown
## Problem
The project has no unit tests, making it:
- Difficult to verify correctness of changes
- Hard to catch regressions
- Risky to refactor code
- Unclear how to validate the implementation

## Expected Behavior
The project should have:
- Unit tests for individual functions
- Integration tests for the full pipeline
- Test data for reproducible testing
- CI/CD integration to run tests automatically

## Current Behavior
No testing infrastructure exists. Changes must be manually verified by running the full pipeline.

## Solution
1. Create a `tests/` directory
2. Add unit tests for:
   - Text preprocessing functions
   - Sentiment labeling functions
   - TF-IDF vectorization
   - Model training and evaluation
3. Use pytest or unittest framework
4. Add test data fixtures
5. Set up GitHub Actions for CI/CD

## Files Affected
- New `tests/` directory
- Add `pytest` to `requirements.txt`
```

---

## Issue 16: Missing .gitignore File

**Title:** Add .gitignore to exclude unnecessary files

**Body:**
```markdown
## Problem
The repository likely tracks files that shouldn't be in version control:
- `__pycache__/` directories
- `*.pyc` files
- Virtual environment directories
- IDE-specific files
- Output files (e.g., `output/` directory contents)
- Large data files (if they shouldn't be in repo)

## Expected Behavior
A `.gitignore` file should exclude:
- Python cache files
- Virtual environments
- IDE files (.vscode, .idea, etc.)
- Output/generated files
- OS-specific files (.DS_Store, Thumbs.db)

## Current Behavior
Repository may contain unnecessary files that clutter the repo and cause merge conflicts.

## Solution
Create a comprehensive `.gitignore` file for Python projects, including:
- `__pycache__/`
- `*.py[cod]`
- `*.so`
- `.Python`
- `venv/`, `env/`, `.venv/`
- `.vscode/`, `.idea/`
- `*.egg-info/`
- `output/` (or specific output file patterns)
- `.pytest_cache/`

## Files Affected
- New `.gitignore` file
```

---

## Issue 17: README Missing Key Information

**Title:** Enhance README with missing information

**Body:**
```markdown
## Problem
The README is missing several important pieces of information:
- No project structure/architecture overview
- No explanation of what each module does
- Missing information about expected data format
- No performance metrics or results
- No troubleshooting section
- Missing information about output files
- No examples of expected output

## Expected Behavior
README should include:
- Clear project structure
- Description of each module/file
- Data format requirements
- Expected output examples
- Performance benchmarks (if available)
- Troubleshooting common issues
- Citation information (if applicable)

## Current Behavior
Users may struggle to understand the project structure and how components interact.

## Solution
Enhance README with:
1. Project structure section showing file organization
2. Module descriptions explaining each Python file's purpose
3. Data format section with example
4. Output section explaining what files are generated
5. Results/Performance section
6. Troubleshooting section
7. Examples section with sample commands and outputs

## Files Affected
- `README.md`
```

---

## Issue 18: TF-IDF Vectorizer Configuration Not Exposed

**Title:** Make TF-IDF parameters configurable

**Body:**
```markdown
## Problem
The TF-IDF vectorizer in `TF_IDF.py` uses default parameters with no way to customize:
- No max_features parameter
- No ngram_range configuration
- No min_df/max_df parameters
- No way to pass custom stop words

This limits flexibility for experimentation and optimization.

## Expected Behavior
The TF-IDF vectorizer should accept configuration parameters to allow:
- Limiting vocabulary size
- Using n-grams
- Filtering rare/common terms
- Custom stop word lists

## Current Behavior
All TF-IDF parameters are hardcoded, requiring code changes to experiment with different configurations.

## Solution
1. Add parameters to `apply_tfidf()` function
2. Allow passing TF-IDF configuration
3. Provide sensible defaults
4. Document parameters in docstrings

## Files Affected
- `TF_IDF.py`
- `main.py` (to pass parameters)
```

---

## Issue 19: No Model Persistence/Saving

**Title:** Add functionality to save and load trained models

**Body:**
```markdown
## Problem
The trained model is not saved after training, meaning:
- Model must be retrained every time
- Cannot use the model for inference without retraining
- No way to version or share trained models
- TF-IDF vectorizer is also not saved

## Expected Behavior
The application should:
- Save trained models to disk
- Save TF-IDF vectorizer for consistent preprocessing
- Provide functionality to load saved models
- Support model versioning

## Current Behavior
Every run requires full retraining, which is time-consuming and wasteful.

## Solution
1. Add model saving functionality (using joblib or pickle)
2. Save TF-IDF vectorizer alongside the model
3. Add model loading functionality
4. Create a `models/` directory for saved models
5. Add command-line option to load existing model vs. train new one

## Files Affected
- `Model.py`
- `main.py`
- `TF_IDF.py`
- Consider creating a `model_utils.py` for save/load functions
```

---

## Issue 20: Missing Documentation Strings

**Title:** Add docstrings to all functions and modules

**Body:**
```markdown
## Problem
Functions and modules lack documentation strings (docstrings), making it:
- Difficult to understand function purpose
- Hard to know what parameters mean
- Unclear what functions return
- Impossible to generate API documentation

## Expected Behavior
All functions and modules should have:
- Module-level docstrings explaining purpose
- Function docstrings with:
  - Description of what the function does
  - Parameter descriptions
  - Return value descriptions
  - Example usage (if helpful)

## Current Behavior
Code is undocumented, requiring readers to analyze implementation to understand usage.

## Solution
1. Add module docstrings to all Python files
2. Add docstrings to all functions following Google or NumPy style
3. Include parameter types and descriptions
4. Document return values
5. Add usage examples for complex functions

## Files Affected
- All Python files in the project
```

