Source Repository: https://github.com/Dicklesworthstone/llm_aided_ocr

## Changes

-   **Support for Third-Party OpenAI APIs:**  Now compatible with third-party OpenAI APIs, such as one-api and new-api. Customize the `BASE_URL` via the `.env` file.
-   **Asynchronous Processing Toggle:** Introduced an option to enable/disable asynchronous processing, aiding in the management of API rate limits. Controlled by the `ASYNC_API_REQUESTS` setting in the `.env` file.
-   **Retry Mechanism:** Implemented a retry mechanism for API requests, with a default of 3 retries and a 10-second delay between each attempt.
-   **Temporary File Storage for `pdf2image` Conversion:**  `pdf2image` conversion results are now temporarily stored in a `.temp_pdf_images` folder within the current directory.
-   **Skip Image Conversion if Already Exists:** The script will now automatically skip the image conversion step if corresponding images already exist within the `.temp_pdf_images` directory (checks only for the existence of the first image to determine if conversion was previously completed).
-   **Skip Tesseract OCR if Output File Already Exists:** Added logic to bypass Tesseract OCR processing if the raw output file (`filename_raw_ocr_output.txt`) already exists.
-   **Moved Settings to `.env`:**  Configuration settings have been relocated to a `.env` file for easier management.

## New `.env` Settings

-   **`OPENAI_BASE_URL`**="http://XXXX/v1"  (OpenAI base URL. For services like one-api, new-api, etc., append `/v1` to the URL)
-   **`OPENAI_MAX_TOKENS`**=8194 (Maximum token limit for OpenAI requests)
-   **`ASYNC_API_REQUESTS`**=False (Toggle for asynchronous API requests. Useful for managing rate limits)
-   **`INPUT_PDF_FILE_PATH`**=xxxx.pdf (Path to the PDF file you wish to process)
