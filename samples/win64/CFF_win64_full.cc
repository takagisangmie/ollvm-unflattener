#include <stdio.h>
#include <windows.h>

// Forward declarations
void LogMessage(const wchar_t *message);
int Factorial(int n);
BOOL CreateTestFile(const wchar_t *filename, const wchar_t *content);
BOOL ReadTestFile(const wchar_t *filename, wchar_t *buffer, DWORD bufferSize);
void SleepOperation(DWORD milliseconds);
void MutexOperation();
BOOL CheckFileExists(const wchar_t *filename);
void DeleteFileIfExists(const wchar_t *filename);

// Recursive factorial function
int Factorial(int n) {
  if (n <= 1) {
    return 1;
  }
  return n * Factorial(n - 1);
}

// File creation function
BOOL CreateTestFile(const wchar_t *filename, const wchar_t *content) {
  HANDLE hFile = CreateFileW(filename, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
                             FILE_ATTRIBUTE_NORMAL, NULL);

  if (hFile == INVALID_HANDLE_VALUE) {
    return FALSE;
  }

  DWORD bytesWritten;
  BOOL result =
      WriteFile(hFile, content, lstrlenW(content), &bytesWritten, NULL);

  CloseHandle(hFile);
  return result;
}

// File reading function
BOOL ReadTestFile(const wchar_t *filename, wchar_t *buffer, DWORD bufferSize) {
  HANDLE hFile = CreateFileW(filename, GENERIC_READ, FILE_SHARE_READ, NULL,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

  if (hFile == INVALID_HANDLE_VALUE) {
    return FALSE;
  }

  DWORD bytesRead;
  BOOL result = ReadFile(hFile, buffer, bufferSize - 1, &bytesRead, NULL);

  if (result) {
    buffer[bytesRead] = '\0';
  }

  CloseHandle(hFile);
  return result;
}

// Sleep operation function
void SleepOperation(DWORD milliseconds) { Sleep(milliseconds); }

// Mutex operation function
void MutexOperation() {
  HANDLE hMutex = CreateMutexW(NULL, FALSE, L"OllvmTestMutex");
  if (hMutex != NULL) {
    DWORD waitResult = WaitForSingleObject(hMutex, 1000);
    if (waitResult == WAIT_OBJECT_0) {
      // Critical section
      LogMessage(L"Mutex acquired");
      SleepOperation(100);
      ReleaseMutex(hMutex);
      LogMessage(L"Mutex released");
    }
    CloseHandle(hMutex);
  }
}

// Logging function
void LogMessage(const wchar_t *message) {
  SYSTEMTIME st;
  GetLocalTime(&st);

  wchar_t timestamp[100];
  wsprintfW(timestamp, L"[%02d:%02d:%02d.%03d] %s\r\n", st.wHour, st.wMinute,
            st.wSecond, st.wMilliseconds, message);

  OutputDebugStringW(timestamp);

  HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
  DWORD written;
  WriteConsole(hConsole, timestamp, lstrlenW(timestamp), &written, NULL);
}

// File existence check
BOOL CheckFileExists(const wchar_t *filename) {
  DWORD attrib = GetFileAttributesW(filename);
  return (attrib != INVALID_FILE_ATTRIBUTES &&
          !(attrib & FILE_ATTRIBUTE_DIRECTORY));
}

// Delete file if it exists
void DeleteFileIfExists(const wchar_t *filename) {
  if (CheckFileExists(filename)) {
    DeleteFileW(filename);
    LogMessage(L"File deleted");
  }
}

// Target function to be obfuscated by OLLVM
void TargetFunction() {
  LogMessage(L"Starting target_function");

  // Variables initialization
  const wchar_t *testFilename = L"ollvm_test.txt";
  const wchar_t *backupFilename = L"ollvm_backup.txt";
  int numbers[] = {1, 2, 3, 4, 5, 6, 7, 8};
  int numbersSize = 8;
  int sum = 0;
  wchar_t buffer[1024];
  wchar_t messageBuffer[256];

  // Conditional statements
  if (CheckFileExists(testFilename)) {
    LogMessage(L"Test file already exists");
    DeleteFileIfExists(testFilename);
  } else {
    LogMessage(L"Test file does not exist yet");
  }

  // For loop
  for (int i = 0; i < numbersSize; i++) {
    sum += numbers[i];
    if (i % 2 == 0) {
      wsprintfW(messageBuffer, L"Processing even index: %d", i);
      LogMessage(messageBuffer);
    } else {
      wsprintfW(messageBuffer, L"Processing odd index: %d", i);
      LogMessage(messageBuffer);
    }
  }

  // Create a test file
  wsprintfW(buffer, L"This is a test file for OLLVM. Sum calculated: %d", sum);
  if (CreateTestFile(testFilename, buffer)) {
    LogMessage(L"Created test file successfully");
  } else {
    LogMessage(L"Failed to create test file");
    return;
  }

  // Sleep operation
  LogMessage(L"Sleeping for 500ms");
  SleepOperation(500);

  // While loop with file operations
  int retryCount = 0;
  while (retryCount < 3) {
    if (ReadTestFile(testFilename, buffer, sizeof(buffer))) {
      LogMessage(L"Read file successfully");
      break;
    }
    LogMessage(L"Failed to read file, retrying");
    retryCount++;
    SleepOperation(100);
  }

  // Do-while loop with factorial calculation
  int factInput = 5;
  int factResult = 0;
  do {
    factResult = Factorial(factInput);
    wsprintfW(messageBuffer, L"Factorial of %d is %d", factInput, factResult);
    LogMessage(messageBuffer);
    factInput--;
  } while (factInput > 0);

  // Mutex operation
  MutexOperation();

  // Final cleanup
  if (CheckFileExists(testFilename)) {
    // Create backup before deleting
    if (ReadTestFile(testFilename, buffer, sizeof(buffer))) {
      if (CreateTestFile(backupFilename, buffer)) {
        LogMessage(L"Created backup file");
      }
    }
    DeleteFileIfExists(testFilename);
  }

  LogMessage(L"Completed target_function");
}

// Main function
int main(int argc, char *argv[]) {
  LogMessage(L"Program started");

  TargetFunction();

  LogMessage(L"Program completed successfully");
  return 0;
}