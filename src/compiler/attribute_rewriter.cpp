#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct source_file {
  std::vector<char> contents;
};

std::unordered_map<int, source_file> &get_file_cache() {
  static std::unordered_map<int, source_file> file_cache;
  return file_cache;
}

int shift_lambda_attribute(std::vector<char> &buffer) {
  int count = 0;

  constexpr int max_attribute_length = 100;

  std::size_t i = 0;
  bool in_string = false;
  while (i < buffer.size() - 1) {
    // Toggle inString flag
    if (buffer[i] == '"' && buffer[i - 1] != '\\') {
      in_string = !in_string;
    }

    // Skip if in string literal
    if (in_string) {
      i++;
      continue;
    }

    if (buffer[i] == '[' && buffer[i + 1] == '[') {
      const std::size_t start = i;
      i += 2;

      while (i < buffer.size() - 1 && (i - start) < max_attribute_length &&
             !(buffer[i] == ']' && buffer[i + 1] == ']'))
        i++;
      if (i >= buffer.size() - 1 || (i - start) >= max_attribute_length)
        continue;

      const std::size_t end = i + 2;
      const std::string attr(buffer.data() + start, end - start);

      if (attr.find("sycl::") == std::string::npos)
        continue;

      if (attr.find("(") == std::string::npos)
        continue;

      if (attr.find(")") == std::string::npos)
        continue;

      if (attr.find("()") != std::string::npos)
        continue;

      // Check that the previous token is not operator.
      bool shouldUseStandardSyntax = [&] {
        std::size_t j = start;
        while (j > 0 && (buffer[j - 1] == ' ' || buffer[j - 1] == '\n' ||
                         buffer[j - 1] == '\r' || buffer[j - 1] == '\t' ||
                         buffer[j - 1] == '(' || buffer[j - 1] == ')'))
          j--;
        if (j >= 8) {
          std::string prev_token(buffer.data() + j - 8, 8);
          return (prev_token == "operator");
        }
        return false;
      }();

      // Replace the attribute with an annotate attribute.
      std::string annotate_attr;
      if (shouldUseStandardSyntax)
        annotate_attr = "[[clang::annotate(\"";
      else
        annotate_attr = "__attribute__((annotate(\"";

      std::string attrWithoutBrackets = attr.substr(2, attr.size() - 4);

      // Replace the left paren with a comma.
      for (char c : attrWithoutBrackets)
        if (c == '(')
          annotate_attr += "\", ";
        else
          annotate_attr += c;

      if (shouldUseStandardSyntax)
        annotate_attr += "]]";
      else
        annotate_attr += "))";

      // Remove hint attributes.
      if (attr.find("_hint") != std::string::npos)
        annotate_attr = "";

      // Replace in buffer.
      buffer.erase(buffer.begin() + start, buffer.begin() + end);
      buffer.insert(buffer.begin() + start, annotate_attr.begin(),
                    annotate_attr.end());
      count++;
      i = start + annotate_attr.size() - 1;
    }
    i++;
  }

  return count;
}

#if __APPLE__
extern "C" int protosycl_open(const char *pathname, int flags, ...) {
  typedef int (*orig_open_type)(const char *, int, ...);
  static orig_open_type orig_open = open;
#else
extern "C" int open(const char *pathname, int flags, ...) {
  typedef int (*orig_open_type)(const char *, int, ...);
  static orig_open_type orig_open = (orig_open_type)dlsym(RTLD_NEXT, "open");
#endif

  // Call the original open to get a file descriptor
  int fd;
  if (flags & O_CREAT) {
    va_list args;
    va_start(args, flags);
    mode_t mode = va_arg(args, int);
    fd = orig_open(pathname, flags, mode);
    va_end(args);
  } else {
    fd = orig_open(pathname, flags);
  }

  // Return if the open failed.
  if (fd == -1)
    return fd;

  // Return if pathname starts with proc.
  if (strncmp(pathname, "/proc/", 6) == 0)
    return fd;

  if (std::string{pathname}.ends_with(".cpp") == false &&
      std::string{pathname}.ends_with(".hpp") == false &&
      std::string{pathname}.ends_with(".h") == false)
    return fd;

  // Load the full file contents.
  auto contents = [&] {
    off_t file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    std::vector<char> buffer(file_size);
    pread(fd, buffer.data(), file_size, 0);
    lseek(fd, 0, SEEK_SET);
    return buffer;
  }();

  std::size_t original_size = contents.size();

  // Return if the file is empty,
  if (contents.size() == 0)
    return fd;

  // Process the file contents to shift lambda attributes.
  int shifted = shift_lambda_attribute(contents);

  // Return if no attributes were shifted.
  if (shifted == 0)
    return fd;

  // Store the contents in the cache,
  get_file_cache()[fd] = source_file{std::move(contents)};

  // Return the file descriptor,
  return fd;
}

#ifdef __APPLE__
extern "C" int protosycl_fstat(int fd, struct stat *statbuf) {
  typedef int (*orig_fstat_type)(int, struct stat *);
  static orig_fstat_type orig_fstat = fstat;
#else
extern "C" int fstat(int fd, struct stat *statbuf) {
  typedef int (*orig_fstat_type)(int, struct stat *);
  static orig_fstat_type orig_fstat =
      (orig_fstat_type)dlsym(RTLD_NEXT, "fstat");
#endif

  // Call the original fstat
  int result = orig_fstat(fd, statbuf);

  // Return if fstat failed
  if (result != 0)
    return result;

  // Check if the file descriptor is in the cache
  auto &file_cache = get_file_cache();
  auto it = file_cache.find(fd);
  if (it != file_cache.end()) {
    source_file &file = it->second;
    std::size_t block_size = statbuf->st_blksize;
    statbuf->st_size = file.contents.size();
    statbuf->st_blocks = (file.contents.size() + (block_size - 1)) / block_size;
  }

  // Return the result
  return result;
}

#ifdef __APPLE__
extern "C" ssize_t protosycl_pread(int fd, void *buf, size_t count,
                                   off_t offset) {
  typedef ssize_t (*orig_pread_type)(int, void *, size_t, off_t);
  static orig_pread_type orig_pread = pread;
#else
extern "C" ssize_t pread(int fd, void *buf, size_t count, off_t offset) {
  typedef ssize_t (*orig_pread_type)(int, void *, size_t, off_t);
  static orig_pread_type orig_pread =
      (orig_pread_type)dlsym(RTLD_NEXT, "pread");
#endif

  // Check if the file descriptor is in the cache
  auto &file_cache = get_file_cache();
  auto it = file_cache.find(fd);
  if (it != file_cache.end()) {
    source_file &file = it->second;
    // Adjust offset if it's beyond the file size
    if (offset >= file.contents.size())
      return 0; // EOF
    // Calculate how many bytes we can actually read
    size_t bytes_to_read =
        std::min<size_t>(count, file.contents.size() - offset);
    // Copy the data to the buffer
    std::memcpy(buf, file.contents.data() + offset, bytes_to_read);
    return bytes_to_read;
  }

  // Call the original pread
  return orig_pread(fd, buf, count, offset);
}

#ifdef __APPLE__
extern "C" void *protosycl_mmap(void *addr, size_t length, int prot, int flags,
                                int fd, off_t offset) {
  typedef void *(*orig_mmap_type)(void *, size_t, int, int, int, off_t);
  static orig_mmap_type orig_mmap = mmap;
#else
extern "C" void *mmap(void *addr, size_t length, int prot, int flags, int fd,
                      off_t offset) {
  typedef void *(*orig_mmap_type)(void *, size_t, int, int, int, off_t);
  static orig_mmap_type orig_mmap = (orig_mmap_type)dlsym(RTLD_NEXT, "mmap");
#endif

  // Check if the file descriptor is in the cache.
  auto &file_cache = get_file_cache();
  auto it = file_cache.find(fd);
  if (it != file_cache.end()) {
    source_file &file = it->second;
    // Create an anonymous mapping.
    void *mapped = orig_mmap(addr, length, prot, MAP_ANONYMOUS, -1, 0);
    if (mapped == MAP_FAILED)
      return MAP_FAILED;

    // Copy the file contents to the mapped region.
    std::size_t bytes_to_copy =
        std::min<size_t>(length, file.contents.size() - offset);
    std::memcpy(mapped, file.contents.data() + offset, bytes_to_copy);

    // Return the mapped region.
    return mapped;
  }

  // Call the original mmap.
  return orig_mmap(addr, length, prot, flags, fd, offset);
}

#ifdef __APPLE__
extern "C" int protosycl_close(int fd) {
  typedef int (*orig_close_type)(int);
  static orig_close_type orig_close = close;
#else
extern "C" int close(int fd) {
  typedef int (*orig_close_type)(int);
  static orig_close_type orig_close =
      (orig_close_type)dlsym(RTLD_NEXT, "close");
#endif

  // Remove from cache.
  get_file_cache().erase(fd);
  // Call the original close.
  return orig_close(fd);
}

#if __APPLE__
#define DYLD_INTERPOSE(_replacement, _replacee)                                \
  __attribute__((used)) static struct {                                        \
    const void *replacement;                                                   \
    const void *replacee;                                                      \
  } _interpose_##_replacee __attribute__((section("__DATA,__interpose"))) = {  \
      (const void *)(unsigned long)&_replacement,                              \
      (const void *)(unsigned long)&_replacee};

DYLD_INTERPOSE(protosycl_fstat, fstat);
DYLD_INTERPOSE(protosycl_open, open);
DYLD_INTERPOSE(protosycl_pread, pread);
DYLD_INTERPOSE(protosycl_mmap, mmap);
DYLD_INTERPOSE(protosycl_close, close);
#endif
