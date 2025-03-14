// REQUIRES: xptifw, opencl, target-spir
// RUN: %build_collector
// RUN: %{build} -O2 -o %t.opt.out
// RUN: env XPTI_TRACE_ENABLE=1 XPTI_FRAMEWORK_DISPATCHER=%xptifw_dispatcher XPTI_SUBSCRIBERS=%t_collector.dll %{run} %t.opt.out | FileCheck %s --check-prefixes=CHECK,CHECK-OPT
// RUN: %{build} -fno-sycl-dead-args-optimization -o %t.noopt.out
// RUN: env XPTI_TRACE_ENABLE=1 XPTI_FRAMEWORK_DISPATCHER=%xptifw_dispatcher XPTI_SUBSCRIBERS=%t_collector.dll %{run} %t.noopt.out | FileCheck %s --check-prefixes=CHECK,CHECK-NOOPT

#ifdef XPTI_COLLECTOR

#include "../Inputs/memory_info_collector.cpp"

#else

#include <sycl/detail/core.hpp>
#include <sycl/specialization_id.hpp>
#include <sycl/usm.hpp>

using namespace sycl::access;
constexpr sycl::specialization_id<int> int_id(42);

class Functor1 {
public:
  Functor1(short X_,
           sycl::accessor<int, 1, mode::read_write, target::device> &Acc_)
      : X(X_), Acc(Acc_) {}

  void operator()() const { Acc[0] += X; }

private:
  short X;
  sycl::accessor<int, 1, mode::read_write, target::device> Acc;
};

class Functor2 {
public:
  Functor2(short X_,
           sycl::accessor<int, 1, mode::read_write, target::device> &Acc_)
      : X(X_), Acc(Acc_) {}

  void operator()(sycl::id<1> id = 0) const { Acc[id] += X; }

private:
  short X;
  sycl::accessor<int, 1, mode::read_write, target::device> Acc;
};

int main() {
  bool MismatchFound = false;
  sycl::queue Queue{};

  // CHECK:{{[0-9]+}}|Create buffer|[[BUFFERID:[0-9,a-f,x]+]]|0x0|{{i(nt)*}}|4|1|{5,0,0}|{{.*}}.cpp:[[# @LINE + 1]]:24
  sycl::buffer<int, 1> Buf(5);
  sycl::range<1> Range{Buf.size()};
  sycl::nd_range<1> NDRange{Buf.size(), Buf.size()};
  short Val = Buf.size();
  auto PtrDevice = sycl::malloc_device<int>(7, Queue);
  auto PtrShared = sycl::malloc_shared<int>(8, Queue);
  Queue
      .submit([&](sycl::handler &cgh) {
        // CHECK: {{[0-9]+}}|Construct accessor|[[BUFFERID]]|[[ACCID1:.+]]|2014|1026|{{.*}}.cpp:[[# @LINE + 1]]:23
        auto A1 = Buf.get_access<mode::read_write>(cgh);
        // CHECK: {{[0-9]+}}|Construct accessor|0x0|[[ACCID2:.*]]|2016|1026|{{.*}}.cpp:[[# @LINE + 1]]:38
        sycl::local_accessor<int, 1> A2(Range, cgh);
        cgh.parallel_for<class FillBuffer>(NDRange, [=](sycl::nd_item<1> ndi) {
          auto gid = ndi.get_global_id(0);
          // CHECK-OPT: arg0 : {1, {{[0-9,a-f,x]+}}, 2, 0}
          int h = Val;
          // CHECK-OPT: arg1 : {1, {{.*}}0, 20, 1}
          A2[gid] = h;
          // CHECK-OPT: arg2 : {0, [[ACCID1]], 4062, 2}
          // CHECK-OPT: arg3 : {1, [[ACCID1]], 8, 3}
          A1[gid] = A2[gid];
          // CHECK-OPT: arg4 : {3, {{.*}}, 8, 4}
          PtrDevice[gid] = gid;
          // CHECK-OPT: arg5 : {3, {{.*}}, 8, 5}
          PtrShared[gid] = PtrDevice[gid];
        });
      })
      .wait();

  // CHECK: Wait begin|{{.*}}.cpp:[[# @LINE + 2]]:9
  // CHECK: Wait end|{{.*}}.cpp:[[# @LINE + 1]]:9
  Queue.wait();

  // CHECK: {{[0-9]+}}|Construct accessor|[[BUFFERID]]|[[ACCID3:.*]]|2018|1024|{{.*}}.cpp:[[# @LINE + 1]]:25
  { sycl::host_accessor HA(Buf, sycl::read_only); }

  Queue.submit([&](sycl::handler &cgh) {
    // CHECK: {{[0-9]+}}|Construct accessor|[[BUFFERID]]|[[ACCID4:.+]]|2014|1026|{{.*}}.cpp:[[# @LINE + 1]]:20
    auto Acc = Buf.get_access<mode::read_write>(cgh);
    Functor1 F(Val, Acc);
    // CHECK-OPT: Node create|{{.*}}Functor1|{{.*}}.cpp:[[# @LINE - 4 ]]:9|{1, 1, 1}, {0, 0, 0}, {0, 0, 0}, 3
    // CHECK-NOOPT: Node create|{{.*}}Functor1|{{.*}}.cpp:[[# @LINE - 5 ]]:9|{1, 1, 1}, {0, 0, 0}, {0, 0, 0}, 5
    cgh.single_task(F);
    // CHECK-OPT: arg0 : {1, {{[0-9,a-f,x]+}}, 2, 0}
    // CHECK-OPT: arg1 : {0, [[ACCID4]], 4062, 1}
    // CHECK-OPT: arg2 : {1, [[ACCID4]], 8, 2}
  });

  Queue.submit([&](sycl::handler &cgh) {
    // CHECK: {{[0-9]+}}|Construct accessor|[[BUFFERID]]|[[ACCID5:.+]]|2014|1026|{{.*}}.cpp:[[# @LINE + 1]]:20
    auto Acc = Buf.get_access<mode::read_write>(cgh);
    Functor2 F(Val, Acc);
    // CHECK-OPT: Node create|{{.*}}Functor2|{{.*}}.cpp:[[# @LINE - 4 ]]:9|{5, 1, 1}, {0, 0, 0}, {0, 0, 0}, 3
    // CHECK-NOOPT: Node create|{{.*}}Functor2|{{.*}}.cpp:[[# @LINE - 5 ]]:9|{5, 1, 1}, {0, 0, 0}, {0, 0, 0}, 5
    cgh.parallel_for(Range, F);
    // CHECK-OPT: arg0 : {1, {{[0-9,a-f,x]+}}, 2, 0}
    // CHECK-OPT: arg1 : {0, [[ACCID5]], 4062, 1}
    // CHECK-OPT: arg2 : {1, [[ACCID5]], 8, 2}
  });

  return 0;
}
#endif
