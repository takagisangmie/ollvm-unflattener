#include <stdio.h>
#include <string.h>
#include <windows.h>

// Recursive factorial function
int factorial(int n)
{
  if (n <= 1)
  {
    return 1;
  }
  return n * factorial(n - 1);
}

// Function to check if file exists
bool check_file_exists(const char *filename)
{
  DWORD attributes = GetFileAttributesA(filename);
  return (attributes != INVALID_FILE_ATTRIBUTES &&
          !(attributes & FILE_ATTRIBUTE_DIRECTORY));
}

// Function that creates a mutex
HANDLE create_named_mutex(const char *mutex_name)
{
  HANDLE hMutex = CreateMutexA(NULL, FALSE, mutex_name);
  if (hMutex == NULL)
  {
    printf("CreateMutex error: %lu\n", GetLastError());
    return NULL;
  }

  if (GetLastError() == ERROR_ALREADY_EXISTS)
  {
    printf("Mutex already exists!\n");
  }

  return hMutex;
}

// Function to write data to file
bool write_to_file(const char *filename, const char *data)
{
  HANDLE hFile = CreateFileA(filename, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
                             FILE_ATTRIBUTE_NORMAL, NULL);

  if (hFile == INVALID_HANDLE_VALUE)
  {
    printf("Could not create file: %lu\n", GetLastError());
    return false;
  }

  DWORD bytesWritten;
  BOOL result = WriteFile(hFile, data, strlen(data), &bytesWritten, NULL);

  CloseHandle(hFile);
  return result != 0;
}

// Target function with various control structures and API calls
void target_function(int iterations)
{
  printf("Starting target function with %d iterations\n", iterations);

  // Create a mutex
  HANDLE hMutex = create_named_mutex("OllvmTestMutex");

  // Loop with conditional branching
  for (int i = 0; i < iterations; i++)
  {
    printf("Iteration %d of %d\n", i + 1, iterations);

    if (i % 3 == 0)
    {
      printf("Computing factorial of %d: %d\n", i, factorial(i));
    }
    else if (i % 3 == 1)
    {
      printf("Sleeping for %d milliseconds\n", i * 100);
      Sleep(i * 100);
    }
    else
    {
      const char *filename = "ollvm_test.txt";
      char buffer[100];
      sprintf(buffer, "Data from iteration %d\n", i);

      if (write_to_file(filename, buffer))
      {
        printf("Successfully wrote to file\n");
      }

      if (check_file_exists(filename))
      {
        printf("Verified file exists\n");
      }
    }
  }

  // Clean up
  if (hMutex != NULL)
  {
    CloseHandle(hMutex);
  }

  printf("Target function completed\n");
}

int main(int argc, char *argv[])
{
  int iterations = 5;

  if (argc > 1)
  {
    iterations = atoi(argv[1]);
  }

  printf("OLLVM Test Program\n");
  target_function(iterations);

  return 0;
}