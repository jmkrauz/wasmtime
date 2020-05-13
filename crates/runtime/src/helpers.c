#include <setjmp.h>

int RegisterSetjmp(
    void **buf_storage,
    void (*body)(void*),
    void *payload) {
  jmp_buf buf;
  if (setjmp(buf) != 0) {
    return 0;
  }
  *buf_storage = &buf;
  body(payload);
  return 1;
}

void Unwind(void *JmpBuf) {
  jmp_buf *buf = (jmp_buf*) JmpBuf;
  longjmp(*buf, 1);
}

#ifdef __arm__

void call_thumb(void *payload) {
  register void *fun __asm__("r0") = payload;
  
  __asm__ volatile (
    "orr %0, %0, #1\n\t"
    "blx %0"
    : "+g"(fun)
    : "g"(fun)
    : "r1", "r2", "r3", "ip", "lr", "cc", "memory");
}

#endif

#ifdef __APPLE__
#include <sys/ucontext.h>

void* GetPcFromUContext(ucontext_t *cx) {
  return (void*) cx->uc_mcontext->__ss.__rip;
}
#endif

#if defined(__linux__) && defined(__aarch64__)
#include <sys/ucontext.h>

void* GetPcFromUContext(ucontext_t *cx) {
    return (void*) cx->uc_mcontext.pc;
}

#endif  // __linux__ && __aarch64__

#if defined(__linux__) && defined(__arm__)
#include <sys/ucontext.h>

void* GetPcFromUContext(ucontext_t *cx) {
    return (void*) cx->uc_mcontext.arm_pc;
}

#endif // __linux__ && __arm__
