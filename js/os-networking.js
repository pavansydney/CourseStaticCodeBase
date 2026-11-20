// ============================================================
// Operating Systems + Networking curriculum (deep rewrite)
// Parser-safe labels and Mermaid-friendly diagrams.
// ============================================================

/* global courseData */

// Cross-platform paper labs need no elevated privileges or live network traffic.
// ---------- Track 1: OS Foundations ----------
courseData.osFundamentals = [
    {
        number: "OS - Module 1",
        title: "Operating System Architecture and Execution Model",
        description: "Understand what an OS guarantees, how privilege boundaries work, and how software reaches hardware safely.",
        duration: "110 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["OS guarantees", "Kernel and user mode", "System calls", "Interrupts and traps", "Boot sequence"],
        detailedDescription: "This module establishes first-principles OS thinking so later topics like scheduling, memory, and security become intuitive.",
        detailedContent: [
            {
                title: "Lesson 1: What an Operating System Guarantees",
                content: `Learning Objective: Define the core guarantees that operating systems provide to applications and users.
Core Theory: An operating system is a control layer that manages CPU time, memory ownership, device access, and storage abstraction. It enforces isolation boundaries so one process cannot freely corrupt another process. It also provides consistent APIs so applications can run without hardware-specific logic.
Diagram (Mermaid):
flowchart TD
A[Application requests] --> B[OS abstraction layer]
B --> C[CPU scheduling]
B --> D[Memory isolation]
B --> E[File and device I/O]
Worked Example: A process asks to read 2 bytes from handle 3, whose fixture contents are ABC and offset is 0. The kernel checks the handle and writable buffer, copies AB and advances offset to 2, then returns 2. Invalid handle 9 returns an error without advancing handle 3. The API hides device details but still exposes partial reads and errors.
Common Mistakes: Treating the OS as just a launcher instead of a safety and resource-management system.
Recap:
- OS provides abstraction and isolation
- Resource arbitration is continuous
- APIs decouple apps from hardware details
Practice:
- Trace read(3,2) followed by read(3,2) on ABC. Expected return counts are 2 then 1, buffers AB then C, and final offset 3. Explain why a request for 2 bytes need not return 2. Use a paper handle table, not platform-specific syscall numbers.`
            },
            {
                title: "Lesson 2: User Mode vs Kernel Mode",
                content: `Learning Objective: Explain why modern systems separate unprivileged and privileged execution.
Core Theory: User mode restricts privileged instructions and memory access. Kernel mode runs privileged OS code through controlled entry points. A syscall changes privilege but need not switch to another process; blocking or scheduling may cause a separate context switch. Isolation depends on correct hardware and kernel enforcement, not an unconditional guarantee against all bugs.
Diagram (Mermaid):
flowchart LR
A[User process] --> B[Trap or syscall]
B --> C[Kernel handler]
C --> D[Validated operation]
D --> E[Return to user mode]
Worked Example: open() enters kernel mode so permissions and path resolution are enforced before a file descriptor is returned.
Common Mistakes: Assuming application code can directly configure hardware or page tables.
Recap:
- Privilege separation protects stability
- Kernel entry is explicit and audited
- Most app operations stay in user mode
Practice:
- Explain one failure mode if all code ran in kernel mode`
            },
            {
                title: "Lesson 3: System Calls and ABI Boundaries",
                content: `Learning Objective: Understand syscall interfaces and why ABI conventions matter.
Core Theory: A system call is a contract between user space and kernel space. The ABI defines argument passing, return values, and error semantics. Language runtimes and libraries wrap syscalls but do not eliminate kernel boundary rules.
Diagram (Mermaid):
sequenceDiagram
participant App
participant Runtime
participant Kernel
App->>Runtime: read(file)
Runtime->>Kernel: syscall read
Kernel-->>Runtime: bytes or error
Runtime-->>App: return result
Worked Example: A library may satisfy a buffered read from already-cached user-space bytes with no new syscall. When it needs kernel I/O, its platform wrapper uses that OS's ABI. POSIX read commonly returns -1 and sets errno on errors through its wrapper; Windows APIs expose different handles and error conventions. Library calls and syscalls are not one-to-one.
Common Mistakes: Confusing library APIs with kernel APIs.
Recap:
- Syscalls are strict interfaces
- ABI compatibility is required for correctness
- Wrappers simplify usage, not semantics
Practice:
- Name two high-level functions that ultimately require syscalls`
            },
            {
                title: "Lesson 4: Interrupts, Exceptions, and Preemption",
                content: `Learning Objective: Differentiate asynchronous interrupts from synchronous exceptions and connect both to scheduling.
Core Theory: Interrupts originate from external events like timers and devices. Exceptions are triggered by the running instruction stream, such as divide-by-zero or page fault. Timer interrupts enable preemptive scheduling so interactive tasks stay responsive.
Diagram (Mermaid):
flowchart TD
A[Running task] --> B{Event}
B -->|Timer IRQ| C[Scheduler path]
B -->|Device IRQ| D[Driver handler]
B -->|Exception| E[Fault handler]
C --> F[Potential context switch]
Worked Example: A timer interrupt preempts a CPU-heavy process to run a waiting shell command.
Common Mistakes: Treating all exceptions as fatal crashes.
Recap:
- Interrupts and exceptions are different signals
- Preemption depends on timer events
- Fault handlers can recover in many cases
Practice:
- Describe how page faults can be part of normal execution`
            },
            {
                title: "Lesson 5: Boot Flow from Firmware to Services",
                content: `Learning Objective: Trace the startup sequence from power-on to user-space services.
Core Theory: Firmware initializes hardware and loads a bootloader. The bootloader loads kernel plus initial runtime data. Kernel initializes subsystems and launches init/system manager, which starts user-space daemons and service dependencies.
Diagram (Mermaid):
flowchart LR
A[Power on] --> B[Firmware]
B --> C[Bootloader]
C --> D[Kernel init]
D --> E[Init process]
E --> F[User services]
Worked Example: A Linux host boots through UEFI and then starts network, logging, and SSH services via init orchestration.
Common Mistakes: Assuming drivers only load after login.
Recap:
- Boot is staged and dependency-driven
- Kernel and init have distinct roles
- Service startup order affects readiness
Practice:
- Write a six-step boot sequence for a server OS`
            }
        ]
    },
    {
        number: "OS - Module 2",
        title: "Processes, Threads, and CPU Scheduling",
        description: "Learn how execution units are represented, switched, and scheduled under real workload constraints.",
        duration: "120 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Process states", "Threads", "Context switching", "Scheduling policies", "Multicore affinity"],
        detailedDescription: "This module connects process abstractions with practical scheduler behavior, responsiveness, and throughput trade-offs.",
        detailedContent: [
            {
                title: "Lesson 1: Process Lifecycle and PCB",
                content: `Learning Objective: Explain process state transitions and metadata needed for execution management.
Core Theory: Processes move through new, ready, running, blocked, and terminated states. The process control block stores register state, memory references, identifiers, file handles, and accounting data.
Diagram (Mermaid):
flowchart LR
A[New] --> B[Ready]
B --> C[Running]
C --> D[Blocked]
D --> B
C --> E[Terminated]
Worked Example: A worker starts ready, is selected to run, requests an unavailable file page and blocks. When I/O completes it becomes ready, not automatically running; the scheduler must select it again. Its saved instruction pointer and registers allow resumption, while its I/O completion status explains why the wait ended.
Common Mistakes: Thinking blocked tasks still consume CPU time.
Recap:
- State transitions model readiness and progress
- PCB enables pause/resume correctness
- I/O completion drives unblock events
Practice:
- Trace ready -> running -> blocked -> ready -> running for one request. Mark which transition requires scheduler selection and list instruction pointer plus stack pointer as context fields. Expected: I/O completion alone does not immediately run the task.`
            },
            {
                title: "Lesson 2: Threads and Shared Address Space",
                content: `Learning Objective: Compare process and thread models for performance and reliability.
Core Theory: Threads share process memory and resources, making communication fast but increasing shared-state risk. Processes isolate memory by default, reducing blast radius at higher overhead.
Diagram (Mermaid):
flowchart TD
A[Process] --> B[Thread A stack]
A --> C[Thread B stack]
A --> D[Shared heap]
A --> E[Shared descriptors]
Worked Example: A web server uses a thread pool for request handling and a separate process for sandboxed file conversion.
Common Mistakes: Assuming threads are always superior without contention analysis.
Recap:
- Threads reduce boundary overhead
- Processes improve isolation
- Model choice depends on failure tolerance and workload
Practice:
- Choose process vs thread for plugin execution and justify`
            },
            {
                title: "Lesson 3: Context Switching and Overhead",
                content: `Learning Objective: Describe what a context switch changes and why excessive switching hurts performance.
Core Theory: Context switches save current CPU state and load next runnable task state. High switch rates can reduce cache locality and increase scheduler overhead, reducing useful work per second.
Diagram (Mermaid):
flowchart LR
A[Running task] --> B[Save context]
B --> C[Select next task]
C --> D[Load context]
D --> E[Resume]
Worked Example: Tiny timeslices in CPU-bound workloads increase switch overhead and lower throughput.
Common Mistakes: Optimizing for fairness without measuring cache and switch impact.
Recap:
- Context switches are necessary but costly
- Cache effects influence real throughput
- Scheduler tuning requires workload evidence
Practice:
- Explain why microtasks may degrade under overly small quantum`
            },
            {
                title: "Lesson 4: Scheduling Policies and Trade-offs",
                content: `Learning Objective: Map common scheduler algorithms to service goals.
Core Theory: FCFS is simple but can penalize short jobs. Round Robin improves interactive fairness. Priority scheduling handles urgency but risks starvation. Multilevel feedback queues adapt to mixed workloads.
Diagram (Mermaid):
flowchart TD
A[Ready queue] --> B{Policy}
B --> C[FCFS]
B --> D[Round Robin]
B --> E[Priority or MLFQ]
Worked Example: With both jobs ready at time 0, A needs 5 CPU units and B needs 3. Round Robin quantum 2 gives A[0,2), B[2,4), A[4,6), B[6,7), A[7,8). Completion times are A=8, B=7; waiting times completion-burst are 3 and 4. Assume no I/O or switching cost; real policies differ.
Common Mistakes: Using fixed priority without aging or fairness safeguards.
Recap:
- No universal best scheduler
- Policies reflect product priorities
- Starvation prevention is mandatory
Practice:
- Recommend a policy for batch analytics cluster`
            },
            {
                title: "Lesson 5: Multicore Scheduling and CPU Affinity",
                content: `Learning Objective: Explain load balancing and affinity effects in multicore systems.
Core Theory: Schedulers balance runnable tasks across cores while preserving affinity for cache locality. Over-pinning can cause imbalance; under-pinning can increase cache misses for latency-sensitive tasks.
Diagram (Mermaid):
flowchart LR
A[Global workload] --> B[Per-core queues]
B --> C[Core 0]
B --> D[Core 1]
B --> E[Core N]
Worked Example: Pinning a low-latency audio thread reduces jitter, while leaving worker pools load-balanced preserves throughput.
Common Mistakes: Pinning everything and preventing healthy balancing.
Recap:
- Affinity is a tuning lever, not default rule
- Balance and locality are competing goals
- Measure latency and throughput together
Practice:
- Give one case where CPU pinning helps and one where it hurts`
            }
        ]
    },
    {
        number: "OS - Module 3",
        title: "Virtual Memory, Allocation, and Memory Pressure",
        description: "Understand translation, paging, allocator behavior, and diagnostics for memory-bound systems.",
        duration: "120 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Virtual memory", "Page tables", "TLB", "Page faults", "Swapping and fragmentation"],
        detailedDescription: "This module turns memory behavior into a measurable model you can use for debugging and performance tuning.",
        detailedContent: [
            {
                title: "Lesson 1: Virtual Address Space and Isolation",
                content: `Learning Objective: Explain how virtual memory isolates processes while simplifying application development.
Core Theory: Each process sees a private virtual address space. MMU and page tables translate virtual addresses to physical frames and enforce read/write/execute permissions.
Diagram (Mermaid):
flowchart LR
A[Virtual address] --> B[Page table lookup]
B --> C[Physical frame]
C --> D[Permission enforcement]
Worked Example: With 1024-byte pages, virtual address 2500 is page 2 plus offset 452. Process A maps page 2 to frame 7, giving physical address 7*1024+452=7620. Process B maps page 2 to frame 3, giving 3524. The identical virtual address is interpreted using different process mappings.
Common Mistakes: Believing virtual addresses are globally unique.
Recap:
- Virtual memory supports isolation and flexibility
- Translation is hardware-assisted
- Permission bits enforce boundaries
Practice:
- Explain why null-page mapping policy improves safety`
            },
            {
                title: "Lesson 2: Paging, TLB, and Access Performance",
                content: `Learning Objective: Connect translation caching behavior to application latency.
Core Theory: Virtual accesses require address translation, often served by a TLB cache. A TLB miss usually triggers a page-table walk (or a software refill on some systems), not necessarily a page fault. A resident permitted mapping can refill the TLB without storage I/O. Missing mappings and permission violations have different fault outcomes.
Diagram (Mermaid):
flowchart TD
A[Memory reference] --> B{TLB hit}
B -->|Yes| C[Fast path]
B -->|No| D[Page walk]
D --> E[TLB refill]
E --> C
Worked Example: Random access over huge datasets increases TLB misses and slows query execution.
Common Mistakes: Profiling only CPU usage and ignoring translation metrics.
Recap:
- TLB is critical to memory performance
- Access pattern strongly affects miss rate
- Translation overhead can dominate hot paths
Practice:
- Suggest one data-layout change to improve TLB locality`
            },
            {
                title: "Lesson 3: Demand Paging and Fault Handling",
                content: `Learning Objective: Distinguish normal page-fault behavior from pathological fault patterns.
Core Theory: Demand paging establishes mappings lazily. A valid first touch can allocate a zero-filled page, map an already cached file page, or read backing storage. On systems with minor/major fault counters, major generally indicates required backing-store I/O, not necessarily a physical spinning disk. A forbidden access can instead terminate the process; not every fault is recoverable.
Diagram (Mermaid):
flowchart LR
A[Access page] --> B{In RAM}
B -->|No| C[Page fault trap]
C --> D[Validate then map or load page]
D --> E[Update mapping]
E --> F[Resume instruction]
Worked Example: First request after deployment is slower due to cold page cache and initial fault burst.
Common Mistakes: Treating all page faults as crash-level errors.
Recap:
- Faults are often expected in demand paging
- Major faults have high latency cost
- Warmup patterns matter for user experience
Practice:
- Design a warmup step to reduce first-request memory faults`
            },
            {
                title: "Lesson 4: Swapping, Thrashing, and Working Set",
                content: `Learning Objective: Identify when memory pressure turns into throughput collapse.
Core Theory: When active working set exceeds RAM, OS evicts pages and may swap anonymous memory. Thrashing occurs when system spends excessive time paging instead of executing useful instructions.
Diagram (Mermaid):
flowchart TD
A[Memory pressure] --> B[Reclaim caches]
B --> C[Swap pages]
C --> D{Pressure persists}
D -->|Yes| E[Thrashing]
Worked Example: Concurrent notebook jobs trigger swap storms, causing unrelated API latency spikes.
Common Mistakes: Adding more threads under thrashing conditions.
Recap:
- Working set fit is critical
- Swap can preserve liveness but raise latency
- Thrashing is a capacity mismatch signal
Practice:
- List two metrics that indicate swap-driven slowdown`
            },
            {
                title: "Lesson 5: Allocators, Leaks, and Fragmentation",
                content: `Learning Objective: Differentiate memory leaks from allocator fragmentation and tuning opportunities.
Core Theory: Allocators optimize for speed and contention via bins or arenas. Memory growth can come from leaks (unreleased live references) or fragmentation (unusable free-space patterns).
Diagram (Mermaid):
flowchart LR
A[Allocation request] --> B[Allocator bins]
B --> C[Block returned]
C --> D[Free and reuse path]
Worked Example: A service shows steady RSS growth despite stable object count due to fragmentation from mixed-size allocations.
Common Mistakes: Labeling all memory growth as leaks without profiling.
Recap:
- Leak and fragmentation are distinct issues
- Allocator strategy impacts footprint and latency
- Profiling is required before tuning decisions
Practice:
- Propose a test plan to separate leak vs fragmentation symptoms`
            }
        ]
    }
];

// ---------- Track 2: OS Reliability and Security ----------
courseData.osProcessesMemory = [
    {
        number: "OS - Module 4",
        title: "Concurrency, Synchronization, and Deadlocks",
        description: "Move from basic thread safety to practical synchronization architecture and diagnostics.",
        duration: "115 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Race conditions", "Lock granularity", "Condition variables", "Deadlocks", "Priority inversion"],
        detailedDescription: "This module teaches robust shared-state design and failure analysis in concurrent systems.",
        detailedContent: [
            {
                title: "Lesson 1: Race Conditions and Visibility",
                content: `Learning Objective: Explain race conditions and memory visibility guarantees in multithreaded code.
Core Theory: A data race involves conflicting concurrent memory accesses without the ordering required by the language's memory model, with at least one write. In C/C++, non-atomic data races cause undefined behavior, not just occasional lost updates. A logical race can exist even with atomic accesses if a multi-step operation is not protected. Mutexes and appropriate atomics provide specific ordering/visibility guarantees; they do not make all outcomes deterministic.
Diagram (Mermaid):
flowchart LR
A[Thread A write] --> B[Shared state]
C[Thread B read] --> B
B --> D{Synchronized}
D -->|No| E[Nondeterministic result]
Worked Example: In a paper interleaving, counter=0; A reads 0, B reads 0, A computes/writes 1, B computes/writes 1. Final value 1 loses one increment. If each complete read-modify-write is protected by the same mutex, the second thread reads 1 and final value is 2. This is a logical trace, not executable racy C/C++ code.
Common Mistakes: Assuming intermittent test passes prove race-free code.
Recap:
- Race bugs are correctness defects
- Visibility and ordering require explicit synchronization
- Timing-dependent behavior is fragile
Practice:
- Give one scenario where atomic operations are required`
            },
            {
                title: "Lesson 2: Mutexes, RW Locks, and Contention",
                content: `Learning Objective: Choose lock strategies based on workload read/write patterns.
Core Theory: Coarse locks simplify correctness but increase contention. Fine-grained locks improve parallelism but raise complexity and deadlock risk. Read-write locks help read-dominant paths.
Diagram (Mermaid):
flowchart TD
A[Shared structure] --> B[Coarse lock]
A --> C[Fine-grained locks]
B --> D[Lower complexity]
C --> E[Higher concurrency]
Worked Example: Per-bucket locks in a hash map scale better than one global lock under high concurrency.
Common Mistakes: Splitting locks without documenting lock ordering.
Recap:
- Lock scope influences throughput
- Complexity and performance must be balanced
- Contention metrics guide lock redesign
Practice:
- Recommend lock design for a read-heavy config cache`
            },
            {
                title: "Lesson 3: Condition Variables and Producer-Consumer",
                content: `Learning Objective: Use signaling primitives to coordinate threads without busy waiting.
Core Theory: Condition variables allow threads to sleep until state predicates become true. Correct usage checks predicates in a loop to handle spurious wakeups and races between signal and scheduling.
Diagram (Mermaid):
sequenceDiagram
participant P as Producer
participant Q as Queue
participant C as Consumer
P->>Q: push and signal
C->>Q: wait while empty
Q-->>C: wake and consume
Worked Example: Bounded queue blocks producers when full and consumers when empty, enforcing natural backpressure.
Common Mistakes: Waiting once without predicate re-check.
Recap:
- Signaling avoids wasteful spin loops
- Predicate loops are correctness-critical
- Bounded queues model realistic flow control
Practice:
- Define queue state conditions for wait and notify`
            },
            {
                title: "Lesson 4: Deadlock Detection and Prevention",
                content: `Learning Objective: Apply structured deadlock prevention strategies.
Core Theory: Deadlock requires mutual exclusion, hold-and-wait, no preemption, and circular wait. Prevention approaches include global lock ordering, lock timeout plus rollback, and resource hierarchy design.
Diagram (Mermaid):
flowchart LR
A[Thread 1 holds Lock A] --> B[Waits Lock B]
C[Thread 2 holds Lock B] --> D[Waits Lock A]
B --> E[Circular wait]
D --> E
Worked Example: Standardizing lock acquisition order across modules removes circular wait risk.
Common Mistakes: Adding lock timeouts without safe rollback plan.
Recap:
- Deadlock is structural and preventable
- Ordering rules are simple and effective
- Recovery design is part of prevention strategy
Practice:
- Write an ordering policy for three shared resources`
            },
            {
                title: "Lesson 5: Starvation and Priority Inversion",
                content: `Learning Objective: Distinguish fairness failures from deadlocks and apply mitigations.
Core Theory: Starvation delays progress indefinitely despite system activity. Priority inversion occurs when high-priority tasks wait behind low-priority lock holders while medium-priority work keeps running.
Diagram (Mermaid):
flowchart TD
A[High priority task] --> B[Blocked by low priority lock owner]
B --> C[Medium tasks preempt low owner]
C --> D[Priority inversion]
Worked Example: Audio thread misses deadlines until priority inheritance is enabled.
Common Mistakes: Treating inversion as harmless scheduling noise.
Recap:
- Fairness failures can break latency objectives
- Priority inheritance can reduce inversion
- Scheduler and locking policies interact tightly
Practice:
- Describe one workload that benefits from priority inheritance`
            }
        ]
    },
    {
        number: "OS - Module 5",
        title: "File Systems, Durability, and Storage I/O",
        description: "Understand how filesystem semantics, caching, and journaling impact reliability and performance.",
        duration: "115 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["VFS", "Inodes", "Page cache", "Journaling", "I/O latency"],
        detailedDescription: "This module focuses on the data path from application writes to persistent media and failure recovery behavior.",
        detailedContent: [
            {
                title: "Lesson 1: VFS and Filesystem Abstraction",
                content: `Learning Objective: Explain how one file API supports many filesystem implementations.
Core Theory: Virtual File System provides uniform operations while backend filesystems implement metadata layout, allocation, and consistency behavior.
Diagram (Mermaid):
flowchart TD
A[Application open/read/write] --> B[VFS interface]
B --> C[Specific filesystem driver]
C --> D[Block device]
Worked Example: In a POSIX-like paper filesystem, directory name draft maps to inode 42. An open descriptor references inode 42; renaming draft to final changes the name mapping, not the descriptor's file identity. Cross-filesystem moves and Windows sharing/delete modes can behave differently; do not generalize this trace to every filesystem.
Common Mistakes: Assuming identical durability semantics across all filesystems.
Recap:
- VFS decouples API from implementation
- Filesystem semantics differ in edge cases
- Portability requires explicit durability assumptions
Practice:
- Start with draft -> inode 42 and descriptor 3 -> inode 42, then rename to final on the same modeled filesystem. Expected: draft lookup fails, final resolves to 42, descriptor 3 still references 42. Explain why this identity trace does not prove crash durability.`
            },
            {
                title: "Lesson 2: Inodes, Directories, and Path Lookup",
                content: `Learning Objective: Trace how a path resolves to stored data.
Core Theory: Directory entries map names to inode IDs. Inodes store metadata and data pointers/extents. Path lookup enforces permission checks at each traversal step.
Diagram (Mermaid):
flowchart LR
A[/var/log/app.log] --> B[root dir]
B --> C[var inode]
C --> D[log inode]
D --> E[file inode]
E --> F[data blocks]
Worked Example: Renaming a file within same filesystem often updates directory metadata without moving content blocks.
Common Mistakes: Treating filename as file identity.
Recap:
- Names resolve to inode identity
- Permissions apply throughout traversal
- Rename and move semantics depend on filesystem boundaries
Practice:
- Explain why open descriptors can survive a rename`
            },
            {
                title: "Lesson 3: Page Cache, Writeback, and fsync",
                content: `Learning Objective: Understand delayed-write behavior and explicit durability boundaries.
Core Theory: A successful buffered write commonly reaches memory before durable media. POSIX fsync requests synchronization of a file's data and required metadata; creating or renaming a file may also require synchronizing its containing directory. Windows uses different APIs such as FlushFileBuffers. Guarantees depend on filesystem, remote-storage semantics and honest hardware flush behavior; check errors and document assumptions.
Diagram (Mermaid):
flowchart LR
A[write syscall] --> B[Dirty page cache]
B --> C[Background flush]
C --> D[Persistent media]
Worked Example: Audit event appears successful in app logs but is lost after crash because no durability barrier was used.
Common Mistakes: Equating syscall success with physical persistence.
Recap:
- Caching improves speed but delays durability
- Critical records need explicit sync policy
- Durability strategy must match business risk
Practice:
- Design a safe append-log persistence policy`
            },
            {
                title: "Lesson 4: Journaling and Crash Recovery",
                content: `Learning Objective: Describe what journaling protects and where extra safeguards are needed.
Core Theory: Journaling records metadata intents before full commit, improving structural recovery. It does not automatically guarantee application-level transaction atomicity.
Diagram (Mermaid):
flowchart TD
A[Metadata change] --> B[Journal write]
B --> C[Journal commit]
C --> D[Main structure update]
Worked Example: After power loss, journal replay restores directory consistency, but partial app record may still require app-level recovery.
Common Mistakes: Assuming journaling eliminates all data-loss scenarios.
Recap:
- Journaling improves filesystem integrity recovery
- App-level atomicity still needs careful design
- Recovery behavior varies by mode and implementation
Practice:
- Give one scenario journaling handles well and one it does not`
            },
            {
                title: "Lesson 5: Storage Latency and Tail Behavior",
                content: `Learning Objective: Relate storage queues and device behavior to p95 and p99 latency.
Core Theory: Average latency can hide severe tail spikes. Queue depth, flush storms, and device garbage collection affect long-tail response times.
Diagram (Mermaid):
flowchart LR
A[App I/O requests] --> B[Kernel queue]
B --> C[Device scheduling]
C --> D[Latency distribution]
Worked Example: Per-request fsync improves safety but raises p99 latency under burst traffic.
Common Mistakes: Optimizing only average latency metrics.
Recap:
- Tail latency impacts user experience and SLOs
- Queue and flush policy shape p99 strongly
- Benchmark with realistic traffic patterns
Practice:
- Propose an experiment to measure write p99 under burst load`
            }
        ]
    }
];

// ---------- Track 3: Networking Foundations ----------
courseData.networkingFundamentals = [
    {
        number: "Networking - Module 1",
        title: "Layered Networking and Packet Journey",
        description: "Build a practical packet-level model across link, network, transport, and application layers.",
        duration: "110 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Layer model", "Encapsulation", "MAC and ARP", "Routing", "MTU and fragmentation"],
        detailedDescription: "This module explains how packets move across subnets and routers, and where common connectivity failures originate.",
        detailedContent: [
            {
                title: "Lesson 1: TCP/IP Layer Responsibilities",
                content: `Learning Objective: Map protocol responsibilities across practical TCP/IP layers.
Core Theory: Link layer handles local delivery, internet layer handles addressing and routing, transport layer handles endpoint communication semantics, and application layer defines business protocols.
Diagram (Mermaid):
flowchart TD
A[Application] --> B[Transport]
B --> C[Internet]
C --> D[Link]
Worked Example: For HTTPS over HTTP/1.1, an HTTP request is protected by TLS records, carried as TCP bytes inside IP packets and local-link frames. TCP delivers an ordered byte stream, IP selects routed next hops and Ethernet addresses only the current link. HTTP/3 instead uses QUIC over UDP, so HTTPS does not always imply TCP.
Common Mistakes: Memorizing layers without understanding failure boundaries.
Recap:
- Layer boundaries simplify protocol evolution
- Troubleshooting is faster when mapped by layer
- Protocols compose, not replace each other
Practice:
- Label an HTTP/1.1 HTTPS packet stack and identify the first stage needing investigation when TCP connects but certificate hostname validation fails. Expected: HTTP/TLS above TCP, IP below TCP, link framing below IP; investigate TLS, not an invented HTTP status.`
            },
            {
                title: "Lesson 2: Encapsulation and Header Semantics",
                content: `Learning Objective: Trace packet wrapping and unwrapping through sender, router, and receiver paths.
Core Theory: Data is encapsulated with transport, internet, and link headers before transmission. Routers inspect network headers for forwarding; destination host decapsulates all layers.
Diagram (Mermaid):
flowchart LR
A[App payload] --> B[Transport segment]
B --> C[IP packet]
C --> D[Link frame]
D --> E[Wire]
Worked Example: A web request crosses multiple routers while preserving end-to-end transport context.
Common Mistakes: Assuming intermediate routers parse application payload by default.
Recap:
- Encapsulation enables modular transport
- Header scope differs by layer
- Decapsulation is host-end responsibility
Practice:
- Explain which header changes at each router hop`
            },
            {
                title: "Lesson 3: Local Delivery with MAC and ARP",
                content: `Learning Objective: Explain how hosts discover local next-hop addresses.
Core Theory: Ethernet forwarding uses MAC addresses inside a subnet. ARP resolves IPv4 addresses to MAC addresses on local links and caches results for efficiency.
Diagram (Mermaid):
sequenceDiagram
participant HostA
participant Switch
participant HostB
HostA->>Switch: ARP broadcast
Switch->>HostB: ARP request
HostB-->>Switch: ARP reply
Switch-->>HostA: Reply with MAC
Worked Example: Host 192.0.2.10/24 sends to 198.51.100.20 via gateway 192.0.2.1. With an empty neighbor cache, it resolves the gateway's MAC, not the remote server's MAC. The Ethernet destination is the gateway while the IP destination remains 198.51.100.20. These documentation addresses are for a paper trace, not live probes.
Common Mistakes: Expecting ARP to resolve remote internet hosts.
Recap:
- ARP is subnet-local resolution
- MAC forwarding is link scoped
- Cache state affects first-packet latency
Practice:
- Describe behavior when ARP entry expires mid-session`
            },
            {
                title: "Lesson 4: Routing and Default Gateway",
                content: `Learning Objective: Understand hop-by-hop forwarding decisions for off-subnet traffic.
Core Theory: A host chooses a route using destination-prefix matching, normally preferring the longest matching prefix and then route metrics/policy. Off-link traffic uses that route's next hop; a default gateway is used only when no more specific route wins. A destination can be unreachable when no suitable route exists. Each router repeats its own forwarding decision.
Diagram (Mermaid):
flowchart LR
A[Source host] --> B[Default gateway]
B --> C[Transit router]
C --> D[Destination network]
Worked Example: A laptop reaches cloud API through home router, ISP edge, and regional backbone routers.
Common Mistakes: Assuming source host computes full path in advance.
Recap:
- Routing is incremental at each hop
- Prefix matching drives forwarding
- Gateway correctness is foundational
Practice:
- Read a sample route table and identify selected next hop`
            },
            {
                title: "Lesson 5: MTU, Fragmentation, and Path Issues",
                content: `Learning Objective: Diagnose packet-size mismatch issues and explain PMTU behavior.
Core Theory: MTU limits IP packet size on a link. IPv4 routers may fragment when permitted, but drop oversized packets with DF set and normally signal a fragmentation-needed error. IPv6 routers never fragment: they send Packet Too Big; the source can adjust sizes or fragment. PMTU discovery depends on useful feedback or packetization-layer probing. MSS limits TCP payload and differs from MTU.
Diagram (Mermaid):
flowchart TD
A[Packet size chosen] --> B{Fits path MTU}
B -->|Yes| C[Forward]
B -->|No| D[Drop or fragment]
D --> E[Adjust sender MSS]
Worked Example: VPN overlay lowers effective MTU, causing intermittent HTTPS failures until MSS is tuned.
Common Mistakes: Ignoring MTU when investigating handshake intermittency.
Recap:
- MTU mismatches create subtle failures
- PMTU mechanisms improve reliability
- Tunnels often change effective payload limits
Practice:
- Outline a troubleshooting flow for suspected MTU blackhole`
            }
        ]
    },
    {
        number: "Networking - Module 2",
        title: "Addressing, Subnetting, NAT, and IPv6",
        description: "Develop practical fluency in CIDR planning, translation boundaries, and dual-stack migration.",
        duration: "115 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["CIDR", "Subnet design", "NAT", "IPv6 basics", "Dual stack"],
        detailedDescription: "This module connects address mathematics to real-world network design and migration operations.",
        detailedContent: [
            {
                title: "Lesson 1: CIDR and Prefix Capacity",
                content: `Learning Objective: Interpret CIDR notation and estimate subnet capacity.
Core Theory: An IPv4 /p block contains 2^(32-p) addresses when network-aligned. Conventional broadcast subnets through /30 reserve network and broadcast addresses; /31 point-to-point and /32 host routes are exceptions. Providers can reserve additional addresses. IPv6 allocation and multicast/neighbor discovery do not use this IPv4 broadcast subtraction rule.
Diagram (Mermaid):
flowchart LR
A[IP block] --> B[Prefix length]
B --> C[Network scope]
B --> D[Host range]
Worked Example: Splitting 10.0.0.0/24 into four /26 subnets gives network offsets 0,64,128,192. The first block is .0-.63, with conventional usable range .1-.62. Four blocks have 256 raw and 248 conventional usable addresses before extra provider reservations.
Common Mistakes: Miscounting usable addresses in cloud environments.
Recap:
- Prefix math drives allocation planning
- Subnet sizing affects growth and policy
- Reservation rules must be considered
Practice:
- Divide 10.0.0.0/24 into eight conventional broadcast subnets. Expected: /27, starts .0,.32,.64,.96,.128,.160,.192,.224, each with 32 raw and 30 conventional usable addresses. Do not apply the minus-two rule to /31 point-to-point links or IPv6.`
            },
            {
                title: "Lesson 2: Private and Public Address Boundaries",
                content: `Learning Objective: Explain where private addressing works and where public routing is required.
Core Theory: RFC 1918 private IPv4 ranges are 10.0.0.0/8, 172.16.0.0/12 and 192.168.0.0/16; they are not globally routed on the public internet. Publicly reachable services need an appropriate routable edge, but private addresses and NAT are not authorization controls. Firewalls, application permissions and routing policy still determine exposure.
Diagram (Mermaid):
flowchart TD
A[Private hosts] --> B[NAT or edge]
B --> C[Public internet]
Worked Example: Application servers remain private while public ingress endpoint routes approved traffic inward.
Common Mistakes: Granting public IPs to internal-only workloads.
Recap:
- Private space improves control and isolation
- Public exposure should be minimal and intentional
- Edge architecture defines trust boundaries
Practice:
- Design a three-tier subnet model with private data tier`
            },
            {
                title: "Lesson 3: NAT Behavior and Connection Tracking",
                content: `Learning Objective: Understand source translation, destination translation, and return-path mapping.
Core Theory: Source NAT rewrites outbound source addresses; destination NAT maps inbound public endpoints to internal services. Stateful tracking is required so return packets map to original internal flows.
Diagram (Mermaid):
flowchart LR
A[Internal client] --> B[SNAT plus port mapping]
B --> C[External service]
C --> B
B --> A
Worked Example: Thousands of clients share one public IPv4 through distinct translated source ports.
Common Mistakes: Under-sizing NAT capacity for high connection concurrency.
Recap:
- NAT depends on stateful flow tracking
- Translation can become bottleneck under scale
- Capacity planning must include connection cardinality
Practice:
- Estimate NAT session load for 3000 clients with 20 parallel connections`
            },
            {
                title: "Lesson 4: IPv6 Addressing and Neighbor Discovery",
                content: `Learning Objective: Summarize IPv6 addressing model and local neighbor resolution behavior.
Core Theory: IPv6 expands address space dramatically and supports hierarchical allocation. Neighbor Discovery handles local link resolution and reachability signaling for IPv6 nodes.
Diagram (Mermaid):
flowchart LR
A[IPv6 host] --> B[Neighbor discovery]
B --> C[Local next hop]
A --> D[Global routing]
Worked Example: Dual-stack API publishes A and AAAA records so clients can use either transport path.
Common Mistakes: Assuming IPv6 removes need for perimeter controls.
Recap:
- IPv6 changes scale and design options
- Local reachability uses different mechanisms than IPv4 ARP
- Dual-stack operations are common during migration
Practice:
- List checklist items for enabling IPv6 on an existing service`
            },
            {
                title: "Lesson 5: Dual-Stack Rollout Strategy",
                content: `Learning Objective: Plan a safe transition from IPv4-only to dual-stack networking.
Core Theory: Effective migration includes DNS updates, ACL parity, observability parity, canary traffic, and rollback checkpoints. Operational readiness matters more than simple address enablement.
Diagram (Mermaid):
flowchart LR
A[IPv4 baseline] --> B[Dual-stack canary]
B --> C[Policy and monitoring parity]
C --> D[Broader rollout]
Worked Example: Internal clients are migrated first, then public endpoints after validating error and latency budgets.
Common Mistakes: Enabling IPv6 routes without equivalent firewall and telemetry controls.
Recap:
- Dual-stack rollout is operationally staged
- Security and monitoring parity are mandatory
- Canary strategy reduces blast radius
Practice:
- Define go/no-go gates for public dual-stack cutover`
            }
        ]
    }
];

// ---------- Track 4: Protocols, DNS, and Web Delivery ----------
courseData.networkingProtocols = [
    {
        number: "Networking - Module 3",
        title: "Transport Protocols, Reliability, and Performance",
        description: "Understand TCP and UDP deeply, including reliability semantics, congestion behavior, and practical trade-offs.",
        duration: "120 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["TCP handshake", "Retransmission", "Flow vs congestion", "UDP design", "QUIC concepts"],
        detailedDescription: "This module links transport internals to observed latency, throughput, and failure behavior in production systems.",
        detailedContent: [
            {
                title: "Lesson 1: TCP Lifecycle and Connection Cost",
                content: `Learning Objective: Explain TCP setup, teardown, and their latency implications.
Core Theory: TCP uses a three-way handshake to establish synchronized sequence space. Connection teardown uses FIN/ACK exchange. Frequent short-lived connections amplify handshake overhead.
Diagram (Mermaid):
sequenceDiagram
participant Client
participant Server
Client->>Server: SYN
Server-->>Client: SYN-ACK
Client->>Server: ACK
Client->>Server: Data
Client->>Server: FIN
Server-->>Client: ACK then FIN
Client->>Server: ACK
Worked Example: Client SYN seq100 consumes one sequence number. Server SYN-ACK seq500,ack101 is followed by client ACK seq101,ack501. In a simplified 40 ms RTT path, fresh TCP setup costs about one RTT before the usual client request; TLS and server processing add their own work. Reusing an established connection avoids this TCP setup but not all request latency.
Common Mistakes: Ignoring connection setup overhead in high-QPS microservice paths.
Recap:
- Connection lifecycle has measurable cost
- Reuse can reduce repeated handshake latency
- State transitions help debug socket issues
Practice:
- Write the SYN/SYN-ACK/ACK sequence for initial client seq10 and server seq90. Expected final ACK seq11,ack91. Explain that TCP acknowledgment confirms transport receipt, not that a database transaction committed.`
            },
            {
                title: "Lesson 2: Reliable Delivery and Retransmission",
                content: `Learning Objective: Describe how TCP preserves order and correctness under packet loss.
Core Theory: Sequence numbers, acknowledgments, and retransmission logic provide in-order reliable byte streams. Loss recovery preserves correctness but increases latency and can reduce effective throughput.
Diagram (Mermaid):
flowchart LR
A[Segment sent] --> B[Ack received]
B --> C[Advance window]
A --> D{Timeout or dup ack}
D -->|Yes| E[Retransmit]
Worked Example: 1 percent loss on long path causes noticeable API latency increase despite successful responses.
Common Mistakes: Assuming reliability means performance is unaffected by loss.
Recap:
- Reliability is achieved through control feedback
- Loss recovery has latency and throughput costs
- Observability should include retransmission metrics
Practice:
- Describe impact of packet loss on p95 latency`
            },
            {
                title: "Lesson 3: Flow Control and Congestion Control",
                content: `Learning Objective: Distinguish endpoint buffer protection from network congestion protection.
Core Theory: Flow control prevents sender from overrunning receiver buffers. Congestion control adjusts sending rate to protect shared path stability. Effective throughput is limited by both windows.
Diagram (Mermaid):
flowchart TD
A[Sender] --> B[Flow control window]
A --> C[Congestion window]
B --> D[Receiver safety]
C --> E[Network stability]
Worked Example: A fast server is limited by mobile client receive window even when backbone has spare bandwidth.
Common Mistakes: Using the two terms interchangeably during diagnostics.
Recap:
- Flow and congestion control solve different bottlenecks
- Throughput is bounded by minimum effective window
- Correct diagnosis requires separating these effects
Practice:
- Provide one scenario constrained by flow control and one by congestion`
            },
            {
                title: "Lesson 4: UDP Trade-offs and Application Design",
                content: `Learning Objective: Decide when UDP is suitable and what reliability features must be added at app level.
Core Theory: UDP preserves datagram boundaries but supplies no built-in retransmission, ordered delivery or congestion control. Applications may accept loss or implement acknowledgments, deduplication, bounded retries, jitter buffers and forward-error correction. Internet-facing UDP applications still need congestion control and rate limits; choosing UDP alone does not guarantee low latency.
Diagram (Mermaid):
flowchart LR
A[App datagram] --> B[UDP send]
B --> C[Best-effort path]
C --> D[App-level recovery]
Worked Example: Live voice system prioritizes low delay over perfect packet recovery to preserve conversation quality.
Common Mistakes: Porting TCP assumptions directly onto UDP protocols.
Recap:
- UDP favors timeliness and simplicity
- Reliability is optional and application-defined
- Suitability depends on tolerance for loss and reordering
Practice:
- Design a minimal ack scheme for UDP telemetry`
            },
            {
                title: "Lesson 5: QUIC and HTTP/3 in Practice",
                content: `Learning Objective: Summarize why modern web stacks use QUIC for transport evolution.
Core Theory: QUIC runs over UDP with congestion control, loss recovery and TLS-based security. Independent streams avoid TCP's cross-stream delivery head-of-line blocking, but an individual ordered stream still waits for its own missing bytes and shared congestion can slow all streams. Reduced handshake cost depends on connection history; 0-RTT data has replay risks and must not carry unsafe non-idempotent actions.
Diagram (Mermaid):
flowchart LR
A[HTTP/3 request] --> B[QUIC transport]
B --> C[Integrated crypto handshake]
C --> D[Multiplexed streams]
Worked Example: Mobile client experiencing path changes maintains smoother sessions with modern QUIC implementations.
Common Mistakes: Assuming QUIC removes need for congestion control because it uses UDP.
Recap:
- QUIC is a full transport protocol over UDP
- Setup and stream behavior differ from classic TCP plus TLS
- Deployment still requires observability and tuning
Practice:
- Compare one advantage and one challenge of HTTP/3 adoption`
            }
        ]
    },
    {
        number: "Networking - Module 4",
        title: "DNS, TLS, HTTP Lifecycle, and Troubleshooting",
        description: "Tie resolution, encryption, transport, and application behavior into one end-to-end production debugging model.",
        duration: "125 min",
        lessons: "6 lessons",
        isNew: true,
        isLocked: false,
        topics: ["DNS path", "TLS handshake", "HTTP semantics", "Load balancing", "CDN", "Incident playbook"],
        detailedDescription: "This capstone module gives an operationally useful model for diagnosing web outages and latency regressions.",
        detailedContent: [
            {
                title: "Lesson 1: DNS Resolution and Cache Hierarchy",
                content: `Learning Objective: Trace DNS resolution and explain TTL effects on change propagation.
Core Theory: A recursive resolver follows referrals from root to TLD to authoritative servers when required records are not cached; clients usually ask that resolver rather than walk the hierarchy themselves. Cached records age from the time obtained. TTL changes do not retroactively shorten existing cache entries, and negative answers may also be cached. Resolver stale-serving policies and application caches can extend observed behavior.
Diagram (Mermaid):
flowchart LR
A[Client cache] --> B[Recursive resolver]
B --> C[Authoritative chain]
C --> B
B --> A
Worked Example: A record cached at t=0 with TTL 60 remains valid at t=30 even if authority changed at t=20. It has 30 seconds remaining. With ordinary expiry and successful refresh at t=60, the new value becomes visible; lowering TTL at t=20 does not retroactively shorten the cached entry.
Common Mistakes: Debugging HTTP before validating name resolution.
Recap:
- DNS is distributed and cache-heavy
- TTL controls propagation timing
- Resolver behavior affects incident scope
Practice:
- Trace a cached record at t=0, authoritative change at t=20 and queries at t=30 and t=60. Expected: old value at t=30; new value at t=60 only under explicit expiry/refresh assumptions. Record resolver and observation time before diagnosing higher layers.`
            },
            {
                title: "Lesson 2: TLS Handshake and Certificate Trust",
                content: `Learning Objective: Explain key handshake stages and certificate validation outcomes.
Core Theory: TLS negotiates cryptographic parameters, verifies certificate trust chain and host identity, then establishes encrypted session keys.
Diagram (Mermaid):
sequenceDiagram
participant C as Client
participant S as Server
C->>S: ClientHello
S-->>C: ServerHello and certificate
C->>S: Key exchange and verify
S-->>C: Finished
C->>S: Encrypted HTTP data
Worked Example: Service remains reachable on port 443 but fails requests due to expired certificate.
Common Mistakes: Treating all HTTPS failures as network outages.
Recap:
- TLS failure can occur even with healthy transport path
- Certificate lifecycle must be operationally managed
- Handshake telemetry is key for diagnosis
Practice:
- List three checks for TLS certificate incident response`
            },
            {
                title: "Lesson 3: HTTP Semantics and Caching Controls",
                content: `Learning Objective: Apply method semantics and status codes to API correctness and caching design.
Core Theory: Methods define intent and idempotency expectations. Status codes communicate outcome class. Cache headers and validators govern freshness and origin load.
Diagram (Mermaid):
flowchart TD
A[Client request] --> B[Origin or cache]
B --> C[Status and headers]
C --> D[Client behavior]
Worked Example: Conditional GET with ETag returns 304 and avoids payload retransmission.
Common Mistakes: Using GET for mutating operations.
Recap:
- HTTP semantics are part of correctness contract
- Cache policy affects both latency and cost
- Method misuse causes subtle production bugs
Practice:
- Define cache headers for static assets vs user-specific API data`
            },
            {
                title: "Lesson 4: Load Balancers and Health Strategy",
                content: `Learning Objective: Design traffic distribution and health checks for resilient service delivery.
Core Theory: Reverse proxies and load balancers route traffic to healthy upstreams. Health checks should test application readiness, not only TCP port openness.
Diagram (Mermaid):
flowchart LR
A[Clients] --> B[Load balancer]
B --> C[Upstream 1]
B --> D[Upstream 2]
B --> E[Upstream N]
Worked Example: Rolling deploy drains one instance at a time while maintaining request success rate.
Common Mistakes: Sending traffic to nodes that pass port checks but fail dependency checks.
Recap:
- Health checks must reflect real readiness
- Load balancing is an availability control plane
- Deployment strategy affects outage risk
Practice:
- Propose readiness and liveness checks for an API service`
            },
            {
                title: "Lesson 5: CDN and Edge Caching Strategy",
                content: `Learning Objective: Explain how edge caching improves latency and protects origin systems.
Core Theory: CDN serves cacheable content near users and reduces origin traffic. Cache key design and invalidation logic determine correctness under personalization and updates.
Diagram (Mermaid):
flowchart LR
A[Origin server] --> B[CDN edge]
B --> C[Regional users]
Worked Example: Product images served from edge reduce origin load and improve global page load speed.
Common Mistakes: Caching personalized responses with overly broad cache keys.
Recap:
- Edge caching improves performance and resilience
- Key design and invalidation are critical
- CDN behavior must align with content semantics
Practice:
- Define cache-key components for localized static content`
            },
            {
                title: "Lesson 6: End-to-End Incident Troubleshooting Workflow",
                content: `Learning Objective: Use a structured, layer-by-layer method to diagnose web outages and latency spikes.
Core Theory: Effective response starts with scope and blast radius, then tests resolution, connectivity, handshake, protocol errors, and backend saturation in sequence.
Diagram (Mermaid):
flowchart TD
A[Alert received] --> B[Scope and impact]
B --> C[Layered diagnostics]
C --> D[Mitigation]
D --> E[Root cause and prevention]
Worked Example: Sudden 5xx spike traced to exhausted DB connection pool after deployment, mitigated by rollback and pool tuning.
Common Mistakes: Skipping evidence timeline while firefighting.
Recap:
- Structured diagnostics reduce MTTR
- Mitigation and root-cause analysis are separate phases
- Post-incident actions prevent recurrence
Practice:
- Write a one-page runbook for sudden API timeout increase`
            }
        ]
    }
];

(() => {
    const groups = [
        ["osFundamentals", [
            {
                prerequisites: "Bytes, functions, and the distinction between a process and the kernel. A handle identifies an open resource in a process; it is not a physical disk address.",
                setup: "Paper or a plain console with an in-memory handle table. No native syscall invocation, installation, administrator privileges or filesystem changes. The trace is portable; real handle/error conventions are OS-specific.",
                workedExample: {
                    problem: "Simulate reads from handle 3 containing bytes ABC at offset 0, with a valid writable destination buffer.",
                    steps: [
                        "Application requests 2 bytes. A runtime wrapper enters the kernel when I/O is needed. Kernel validates handle and buffer, copies AB, advances offset to 2 and returns count 2.",
                        "The next request asks for 2 bytes but only C remains. Return count 1 and advance offset to 3; another read at EOF returns count 0 in this regular-file model.",
                        "Request on invalid handle 9 returns a modeled invalid-handle error and changes no offsets. Returning from a syscall is a privilege transition; a context switch to another process is not required unless scheduling/blocking calls for it."
                    ],
                    expected: "Counts [2,1,0], data ['AB','C',''], offset 3. Invalid handle has an error with no state change. No real kernel guarantees are inferred from this mock."
                },
                practice: {
                    task: "Implement a bounded read simulator for regular files; reject negative or noninteger byte counts and unknown handles.",
                    cases: [
                        { input: "Fresh ABC at offset 0; read handle 3, count 1, then count 5", expected: "Return A/count 1 then BC/count 2; final offset 3." },
                        { input: "Fresh ABC at offset 0; read handle 9, count 2", expected: "Invalid-handle result; handle 3 offset remains 0." },
                        { input: "Fresh ABC; count=-1", expected: "Validation error; no offset change." }
                    ],
                    hint: "bytesReturned=min(requested,remaining). Preserve offset on errors rather than treating every error as EOF.",
                    solution: "Validate count and handle first, slice data from offset to offset+count, then increase offset only by the returned length. Keep EOF distinct from an error. Pipes, sockets, nonblocking calls and asynchronous Windows APIs have additional semantics excluded from this regular-file model."
                },
                interview: {
                    question: "Is a library read always a syscall, and is a syscall always a context switch?",
                    answer: "Neither. A library can serve buffered data in user space. A syscall enters privileged code but can return to the same thread; a scheduler context switch saves one task and resumes another. Blocking I/O may involve both transitions."
                },
                readiness: ["Trace all three successful counts without reading beyond EOF.", "Show unchanged state on an invalid handle or negative count.", "Distinguish library buffering, privilege change and task switching."]
            },
            {
                prerequisites: "Process ready/running/blocked states and arithmetic over time intervals. Waiting time excludes execution and, in this simplified exercise, there is no I/O.",
                setup: "Use a paper Gantt chart or console simulation, not host scheduler settings. One CPU, all jobs arrive at time 0, zero switch cost; cap simulations at 20 CPU units.",
                workedExample: {
                    problem: "Schedule A with burst 5 and B with burst 3 using Round Robin quantum 2, initially queued A then B.",
                    steps: [
                        "Run A[0,2), B[2,4), A[4,6). Remaining work is A=1 and B=1. Requeue unfinished jobs at the tail.",
                        "Run B[6,7), A[7,8). Completion times are B=7, A=8; turnaround equals completion since arrivals are zero.",
                        "Waiting times are A=8-5=3 and B=7-3=4, average 3.5. First response times are A=0 and B=2, average 1. With FCFS, B first runs at 5; fairness and mean waiting are different goals."
                    ],
                    expected: "Timeline ends at 8 with no idle time, five slices, A wait 3/B wait 4 and responses 0/2. Results apply only to the stated zero-overhead model."
                },
                practice: {
                    task: "Calculate completion, waiting and response for another quantum; reject zero quantum to prevent an infinite simulator loop.",
                    cases: [
                        { input: "A burst=3, B burst=1, RR quantum=1, both arrive 0", expected: "A[0,1), B[1,2), A[2,3), A[3,4); completions A=4/B=2; waits 1/1; responses 0/1." },
                        { input: "A burst=5, B burst=3, FCFS order A,B", expected: "Completions 5/8, waits 0/5, responses 0/5; mean wait 2.5." },
                        { input: "RR quantum=0", expected: "Reject before scheduling; do not loop." }
                    ],
                    hint: "Response is firstStart-arrival, not completion-burst. Verify total executed time equals total burst.",
                    solution: "Keep remaining burst and firstStart per job; consume min(quantum,remaining), advance the clock, then requeue if unfinished. Derive wait=completion-arrival-burst for these no-I/O cases. Smaller quanta can improve initial responsiveness but add overhead in real systems; this model cannot predict an actual OS's performance."
                },
                interview: {
                    question: "Why might a blocked job have high latency but low CPU use?",
                    answer: "It is waiting for I/O, a lock or another event rather than executing. Increasing CPU priority cannot remove the underlying wait. Examine state and wait cause before changing concurrency or scheduling policy."
                },
                readiness: ["Draw the five-slice worked trace and recover all waiting/response times.", "Show why RR improves B's first response but not mean waiting in this example.", "Explain the missing multicore, arrival-time, I/O and switch-cost effects."]
            },
            {
                prerequisites: "Integer division/remainder, process-local address spaces and read/write permissions. A virtual page maps to a frame; a TLB caches translations.",
                setup: "Paper page table or small in-memory array only. Page size is an artificial 1024 bytes. Never allocate memory to induce swapping, change page files, or inspect another process.",
                workedExample: {
                    problem: "Translate virtual address 2500 for process A, with virtual page 2 mapped to physical frame 7 and read-only permission.",
                    steps: [
                        "Compute page=floor(2500/1024)=2 and offset=2500 mod 1024=452. Check the process-specific mapping and requested access permission.",
                        "A read is allowed and translates to 7*1024+452=7620. A TLB miss with this valid resident mapping needs a translation refill, not a storage read.",
                        "A write to the same read-only page raises a protection fault in this model. If process B maps page 2 to frame 3, its read at virtual address 2500 translates to 3524 instead. A not-present valid page needs fault handling; an invalid page must not be fabricated."
                    ],
                    expected: "A read=7620, A write=protection fault, B read=3524. Equal virtual addresses do not imply shared physical storage."
                },
                practice: {
                    task: "Implement checked translation and trace a two-frame FIFO page cache without touching real memory pressure.",
                    cases: [
                        { input: "Page size=1024; virtual address=2048; page 2 -> frame 7, read permitted", expected: "Page 2, offset 0, physical address 7168." },
                        { input: "Page 2 read-only; write at 2500", expected: "Protection fault; no physical write." },
                        { input: "Two initially empty frames; FIFO references [0,1,0,2,1]", expected: "Miss,miss,hit,miss,hit: 3 misses, 2 hits; page 0 evicted on reference 2." }
                    ],
                    hint: "A FIFO hit does not change eviction order. Translation-cache misses and resident-page misses are different counters.",
                    solution: "Validate a nonnegative integer address, compute page and offset, check mapping/presence/permissions, then frame*1024+offset. FIFO keeps insertion order: [0], [0,1], [0,1], [1,2], [1,2]. This cache simulation does not reproduce an OS replacement policy or measure actual page faults."
                },
                interview: {
                    question: "Does a TLB miss mean the process must read from disk?",
                    answer: "No. The page may already be resident with a valid permitted mapping, so a page-table walk can refill the translation cache. A recoverable page fault may allocate a zero page or map cached data; only some faults require backing-store I/O."
                },
                readiness: ["Compute 7620 and 3524 with explicit quotient/remainder.", "Reject writes on read-only pages before using the frame.", "Reproduce FIFO's three misses and distinguish them from TLB misses and protection faults."]
            }
        ]],
        ["osProcessesMemory", [
            {
                prerequisites: "Threads share memory but each has its own execution state. A read-modify-write sequence is not indivisible unless synchronization makes it so.",
                setup: "Use paper interleavings or scripted steps in a console. Do not run intentionally racy native code or deadlock host threads; the exercise is a finite state model independent of platform memory models.",
                workedExample: {
                    problem: "Explain a lost increment and prevent a two-lock circular wait.",
                    steps: [
                        "Start counter=0. Interleave A.read=0, B.read=0, A.write=1, B.write=1. Final counter=1 although two increments were requested.",
                        "Protect each whole increment with mutex M: A reads/writes then unlocks; B then reads 1 and writes 2. Making only individual reads/writes atomic would not protect the compound increment.",
                        "For locks L1 and L2, A holding L1 while waiting for L2 and B holding L2 while waiting for L1 forms a cycle. Require every path to acquire L1 before L2; B cannot hold L2 while waiting for L1 under that rule."
                    ],
                    expected: "Unprotected abstract trace ends at 1; mutex trace ends at 2; universal L1-before-L2 order eliminates this two-lock cycle, not every possible liveness failure."
                },
                practice: {
                    task: "Trace a one-slot producer-consumer queue and write the conditions guarded by its mutex; handle a spurious wakeup correctly.",
                    cases: [
                        { input: "Queue empty; consumer wakes spuriously", expected: "Recheck while queue is empty under the mutex and wait again; no item consumed." },
                        { input: "Queue=[7]; producer wants to add 8; consumer removes 7", expected: "Producer waits while full; consumer removes 7 and signals space; producer rechecks and enqueues 8; size never exceeds 1." },
                        { input: "A requests L1 then L2; B requests L2 then L1", expected: "Reject B's acquisition order in the model before executing the second path." }
                    ],
                    hint: "A notification is not an item or a stored permit. Waiting must atomically release the mutex and reacquire it before checking the predicate.",
                    solution: "Consumer: lock M; while empty wait(notEmpty,M); pop; signal notFull; unlock. Producer: lock M; while size===1 wait(notFull,M); push; signal notEmpty; unlock. Use the same mutex for state checks/updates. Require a single global lock order; timeouts without safe rollback do not repair partially applied operations."
                },
                interview: {
                    question: "How do deadlock, starvation and priority inversion differ?",
                    answer: "Deadlock prevents a set of tasks progressing through mutual waits. Starvation indefinitely delays one task while others progress. Priority inversion delays a high-priority task behind a lower-priority lock holder, potentially worsened by medium-priority work; priority inheritance may bound that interference but does not fix arbitrary deadlocks."
                },
                readiness: ["Enumerate the four lost-update steps and the synchronized result.", "Recheck queue predicates after every wake while holding the mutex.", "Identify a wait cycle and explain the scope and limits of lock ordering."]
            },
            {
                prerequisites: "File descriptors/handles, cached writes versus persistence, and application record boundaries. Names and open file identity are separate concepts in POSIX-like filesystems.",
                setup: "Use a paper crash-recovery table with byte strings; no real writes, renames, forced crashes or disk benchmarks. POSIX fsync/directory semantics below are conceptual, not portable commands; Windows and network filesystems differ.",
                workedExample: {
                    problem: "Recover an append log record with length=3, payload ABC, checksum=sum of payload byte values modulo 256.",
                    steps: [
                        "ABC has byte values 65,66,67, so checksum=(65+66+67) mod 256=198. A record is accepted only when its declared payload and checksum byte are present and match.",
                        "Crash snapshot S1 has length=3 and payload AB only: ignore the incomplete trailing record. S2 has length=3,payload ABC,checksum=198: accept ABC. Checksum 197 fails even though the record is complete.",
                        "A successful application write is not proof that S2 reached durable media. For a critical append, check write completion/short writes, request the platform's durability barrier and check its result before acknowledging durability; journal replay alone does not validate the app record."
                    ],
                    expected: "S1 recovers no record; S2 recovers ABC; corrupted checksum rejects the record. The toy checksum detects this corruption but is weak and is not a cryptographic integrity check."
                },
                practice: {
                    task: "Write a bounded parser that accepts only complete valid records, and explain a crash-safe replacement protocol under explicit POSIX-like assumptions.",
                    cases: [
                        { input: "length=2,payload XY,checksum=177 (88+89)", expected: "Accept XY." },
                        { input: "length=3,payload ABC,checksum=197", expected: "Reject corruption; do not expose ABC as committed." },
                        { input: "Replacement file content synced; rename completed; containing directory not synced", expected: "Do not promise durable name replacement after crash; platform/filesystem guarantees must be checked." }
                    ],
                    hint: "Reject oversized declared lengths before allocating. File-content durability and directory-entry durability are separate concerns.",
                    solution: "Limit length to 1024 bytes for this parser, check available bytes, recompute checksum and stop at the first bad trailing record. Conceptually write a new file in the same directory, sync/check it, atomically rename where supported, then sync/check the directory before acknowledging replacement. Test failure checkpoints on paper; actual crash guarantees require filesystem-specific verification."
                },
                interview: {
                    question: "Why does journaling not make every application write a transaction?",
                    answer: "A filesystem journal may recover metadata structure without preserving complete application records or ordering multiple files as one transaction. Applications still need framing, validation, durability barriers and recovery rules; an API success or rename alone may not guarantee persistence."
                },
                readiness: ["Calculate checksums 198 and 177 and reject an incomplete record.", "Explain why a valid cached record can still be lost on crash.", "Separate file data, directory name persistence and application transaction semantics without destructive tests."]
            }
        ]],
        ["networkingFundamentals", [
            {
                prerequisites: "IPv4 addresses, a local Ethernet link, destination prefixes and encapsulation. MAC addresses identify link endpoints; IP addresses identify routed destinations.",
                setup: "Draw packets on paper. Use documentation-only 192.0.2.0/24 and 198.51.100.0/24 addresses; send no traffic, perform no packet capture and change no routes or MTU settings.",
                workedExample: {
                    problem: "Host 192.0.2.10/24 sends TCP to 198.51.100.20 via gateway 192.0.2.1. Initial TTL=64 and the path has two forwarding routers, with no NAT.",
                    steps: [
                        "Host's routing table selects the gateway because destination is off-link. If no cached neighbor mapping exists, ARP resolves 192.0.2.1 to gateway MAC G, not the remote host to its MAC.",
                        "First frame has source MAC H and destination G, wrapping IP source 192.0.2.10 and destination 198.51.100.20. Each router replaces link framing for its next link and decrements IPv4 TTL; destination receives TTL=62.",
                        "For a 1500-byte IP MTU and minimum 20-byte IPv4 plus 20-byte TCP headers, maximum TCP payload is 1460. A 1501-byte IPv4 packet with DF set cannot cross that link; model a drop plus fragmentation-needed feedback rather than changing live settings."
                    ],
                    expected: "First Ethernet destination is G, routed IP endpoints stay unchanged, final TTL=62, and payload bound is 1460 bytes under the stated no-options assumption."
                },
                practice: {
                    task: "Choose the longest-prefix route, next-hop MAC target and payload limit for fixed packet fixtures.",
                    cases: [
                        { input: "Routes 0.0.0.0/0 via G1 and 198.51.100.0/24 via G2; destination 198.51.100.20", expected: "Choose /24 via G2; resolve G2's local-link address, not the remote destination." },
                        { input: "IPv4 path MTU=1400; minimum IP/TCP headers 20+20 bytes", expected: "TCP payload bound=1360; options/tunnels can reduce the bound." },
                        { input: "TTL=1 at a forwarding router", expected: "Drop instead of forwarding with TTL 0; an ICMP time-exceeded response may be generated." }
                    ],
                    hint: "Choose a route before resolving the next-hop link address. Do not subtract Ethernet header bytes from an IP MTU.",
                    solution: "Match destination prefixes, choose the most specific route, then use route policy/metric for equal specificity. For these no-NAT fixtures retain IP endpoints, replace MAC framing per hop and subtract one TTL per forwarding router. IPv6 uses Neighbor Discovery rather than ARP, and routers do not fragment IPv6 packets."
                },
                interview: {
                    question: "Why does a host not ARP for every internet server it contacts?",
                    answer: "ARP broadcasts stay on a local IPv4 link. For an off-link route the host needs the local next-hop router's MAC; routers independently choose later hops. A more specific route can override the default gateway."
                },
                readiness: ["Draw the first frame with correct MAC and IP destinations.", "Choose G2 over the default route and calculate both payload bounds.", "Distinguish IPv4 DF/fragmentation feedback from IPv6 Packet Too Big behavior."]
            },
            {
                prerequisites: "Binary prefix lengths, address ranges and the off-link packet trace. NAT rewrites endpoint fields and requires matching return-path state; it is not authentication.",
                setup: "Paper address and NAT tables only; no router configuration, scanning or external requests. Use 10.0.0.0/24 internally and documentation addresses 198.51.100.9 and 203.0.113.20 for the imagined edge.",
                workedExample: {
                    problem: "Split 10.0.0.0/24 into four equal conventional IPv4 broadcast subnets and trace one translated connection.",
                    steps: [
                        "Borrow 2 host bits to get /26; block size=64. Networks are .0,.64,.128,.192 and broadcasts .63,.127,.191,.255.",
                        "Each block has 64 raw addresses, normally 62 usable under the stated network/broadcast convention. First usable ranges are .1-.62 and .65-.126; provider reservations can reduce them further.",
                        "Map TCP 10.0.0.10:51000 -> 203.0.113.20:443 to 198.51.100.9:40001 -> 203.0.113.20:443. A reply to public port 40001 matches this active mapping and is translated back to 10.0.0.10:51000."
                    ],
                    expected: "Four nonoverlapping /26 ranges cover the /24; 248 conventional usable addresses total. Return mapping restores the internal destination port/address, not a generic permission to access the host."
                },
                practice: {
                    task: "Calculate subnet boundaries and reverse NAT lookup; contrast IPv4 broadcast rules with IPv6 allocation.",
                    cases: [
                        { input: "Split 10.0.0.0/24 into eight equal conventional broadcast subnets", expected: "/27, 32 raw/30 usable each; network offsets 0,32,64,96,128,160,192,224." },
                        { input: "Reply to 198.51.100.9:40002 when only the shown 40001 mapping exists", expected: "No matching mapping; this model drops the reply rather than inventing an internal recipient." },
                        { input: "IPv6 subnet 2001:db8:1::/64", expected: "64 host/interface bits; do not subtract IPv4 network and broadcast addresses, and use Neighbor Discovery rather than ARP." }
                    ],
                    hint: "Address block size is 2^(32-prefix) for IPv4. State keys include protocol and relevant remote endpoints, depending on NAT behavior.",
                    solution: "Eight blocks need 3 borrowed bits: /27. Enumerate block starts in increments of 32 and use start+31 as broadcast. Reverse only known unexpired mappings in the mock. /31 point-to-point and /32 host routes are exceptions to the usual minus-two rule. Keep firewall policy separate; dual-stack services need equivalent access controls for both families."
                },
                interview: {
                    question: "Does a private address or NAT make a service secure?",
                    answer: "No. It changes reachability/translation, not caller authorization. Forwarded ports, internal attackers or policy mistakes can expose private services. Enforce explicit network and application access controls; IPv6 also needs policy even when NAT is absent."
                },
                readiness: ["Enumerate all four /26 and all eight /27 boundaries without overlaps.", "Reverse a known NAT mapping and reject an unknown one.", "State the IPv4 capacity exceptions and why IPv6 does not use IPv4 broadcast subtraction."]
            }
        ]],
        ["networkingProtocols", [
            {
                prerequisites: "Byte positions, TCP versus UDP and a connection's two independent sequence spaces. A TCP ACK identifies the next expected byte, not the number of packets received.",
                setup: "Paper transport trace or a bounded array simulation; no sockets or live loss injection. Ignore sequence wrap, SACK and implementation-specific timers; buffer at most three mock segments.",
                workedExample: {
                    problem: "Receiver expects byte 100. Segments are seq100:'ABC', seq106:'GHI', then seq103:'DEF'. Trace cumulative ACKs and ordered delivery.",
                    steps: [
                        "Accept ABC at sequence numbers 100-102, deliver ABC and ACK 103, the first missing byte.",
                        "GHI arrives at 106-108 while 103-105 is missing. Buffer GHI within the assumed receive window; ACK remains 103 and deliver no new bytes.",
                        "DEF fills 103-105. Deliver DEFGHI in order, advance ACK to 109. A duplicate ABC changes neither ACK 109 nor the application stream. TCP preserves byte order, not the sender's segment boundaries."
                    ],
                    expected: "ACKs [103,103,109]; total delivered stream ABCDEFGHI exactly once. The trace illustrates ordering, not a measured loss-recovery delay."
                },
                practice: {
                    task: "Trace sequence/ACK arithmetic and a simplified throughput bound; contrast UDP recovery responsibilities.",
                    cases: [
                        { input: "Receiver expects 500; receives 4 contiguous bytes at seq500 then the same segment again", expected: "ACK 504 both times; deliver four bytes total, not eight." },
                        { input: "Receive window=8000 bytes, congestion window=12000 bytes, RTT=0.1 seconds; ignore headers and other limits", expected: "Window bound=min(8000,12000)/0.1=80000 bytes/second; this is an idealized bound, not measured throughput." },
                        { input: "UDP datagrams IDs [1,3,3] with application requiring unique ordered IDs", expected: "Application must detect missing 2 and duplicate 3 and define bounded recovery; UDP alone does neither." }
                    ],
                    hint: "Cumulative ACK cannot skip a gap. Flow control protects receiver buffers; congestion control protects the path.",
                    solution: "Maintain nextExpected and a bounded reorder buffer, append contiguous bytes only, ignore already-delivered duplicates, and ACK nextExpected. Rate is bounded by usable in-flight bytes divided by RTT plus other bottlenecks. UDP applications must supply needed deduplication/recovery and congestion control; QUIC supplies these over UDP but does not remove all stream stalls."
                },
                interview: {
                    question: "Does a TCP ACK prove the application processed or durably saved a request?",
                    answer: "No. It confirms transport receipt of bytes at the peer's TCP stack, not application success or durable commit. A retry after a lost application response may duplicate a side effect unless the application uses its own idempotency/transaction protocol."
                },
                readiness: ["Produce ACKs 103,103,109 and explain the buffered gap.", "Calculate the 80000-byte/second bound without calling it a benchmark.", "Separate TCP receipt, application commit and UDP/QUIC reliability responsibilities."]
            },
            {
                prerequisites: "DNS records/TTL, TCP connection setup, TLS hostname validation and HTTP status codes. Resolution, transport, authentication and application response are separate failure stages.",
                setup: "Use a fixed paper timeline and response fixtures; no external endpoints, privileged captures or certificate bypass. example.test and the supplied IP addresses are labels, not probe targets.",
                workedExample: {
                    problem: "A resolver caches example.test -> 192.0.2.10 at t=0 with TTL 60 seconds. Authority changes to 192.0.2.20 at t=20. Diagnose an HTTPS failure at t=30.",
                    steps: [
                        "At t=30 the cached answer can still be .10 with 30 seconds of TTL remaining. Lowering the authoritative TTL at t=20 would not retroactively change this entry.",
                        "Fixture TCP connection to .10 succeeds, but its certificate covers other.test, not example.test. Stop at hostname validation; do not disable certificate checks or describe it as an HTTP 500.",
                        "At t=60 assume normal expiry, no stale serving and a successful resolver refresh: answer becomes .20. Its certificate matches and HTTP returns 503, so now the evidence points to service readiness/backend behavior rather than DNS or TLS."
                    ],
                    expected: "At t=30: old cached IP and TLS hostname failure. At t=60 under the explicit refresh assumptions: new IP, TLS success, HTTP 503. The two failures require different explanations."
                },
                practice: {
                    task: "Classify response-stage failures and model conditional HTTP caching without disabling security checks.",
                    cases: [
                        { input: "Cached body has ETag v1; conditional GET If-None-Match:v1; origin still v1", expected: "304 Not Modified with no new representation body; client reuses its stored body under applicable cache rules." },
                        { input: "DNS 50 ms, TCP 40 ms, TLS 40 ms, server plus transfer 30 ms; serial cold path", expected: "Total 160 ms in this no-overlap/no-reuse model." },
                        { input: "TCP succeeds; certificate expired before any HTTP response", expected: "TLS-stage failure; do not infer an HTTP status and do not bypass validation." }
                    ],
                    hint: "Build an evidence table with stage, observation, next check and safe stop condition. A warm connection would omit some cold-path work.",
                    solution: "At each stage record the result before moving on: resolve, connect, validate TLS, inspect HTTP status/headers, then inspect application readiness. A 304 depends on a stored representation; do not serve another user's private cached response. Separate readiness from liveness, and avoid restarting healthy processes merely because a downstream dependency is unavailable."
                },
                interview: {
                    question: "Why can DNS cutover and rollback appear partial even after changing the authoritative record?",
                    answer: "Resolvers and applications may retain answers until their existing TTLs expire, and policies such as stale serving can add complexity. A changed TTL does not rewrite remote caches. Record observation time, resolver/cache layer and endpoint certificate before attributing failure to HTTP."
                },
                readiness: ["Calculate the old record's remaining TTL at t=30 and explain the t=60 assumptions.", "Differentiate hostname failure from a later HTTP 503 without unsafe bypasses.", "Compute the 160 ms cold-path budget and explain 304 body reuse and private-cache isolation."]
            }
        ]]
    ];
    const guideKeys = ["prerequisites", "setup", "workedExample", "practice", "interview", "readiness"];
    const sameKeys = (value, keys) => value && typeof value === "object" && !Array.isArray(value)
        && Object.keys(value).length === keys.length && keys.every(key => Object.hasOwn(value, key));
    const text = value => typeof value === "string" && value.trim().length > 0;
    const texts = value => Array.isArray(value) && value.length >= 2 && value.every(text);
    const expectedNumbers = {
        osFundamentals: ["OS - Module 1", "OS - Module 2", "OS - Module 3"],
        osProcessesMemory: ["OS - Module 4", "OS - Module 5"],
        networkingFundamentals: ["Networking - Module 1", "Networking - Module 2"],
        networkingProtocols: ["Networking - Module 3", "Networking - Module 4"]
    };
    for (const [key, guides] of groups) {
        const modules = courseData[key];
        if (!Array.isArray(modules) || modules.length !== guides.length) {
            throw new Error(`OS/network learningGuide collection mismatch: ${key}; expected ${guides.length} modules`);
        }
        guides.forEach((guide, index) => {
            const label = `${key}[${index}]`;
            if (!modules[index] || !sameKeys(guide, guideKeys)
                || !text(guide.prerequisites) || !text(guide.setup)
                || !sameKeys(guide.workedExample, ["problem", "steps", "expected"])
                || !text(guide.workedExample.problem) || !texts(guide.workedExample.steps) || !text(guide.workedExample.expected)
                || !sameKeys(guide.practice, ["task", "cases", "hint", "solution"])
                || !text(guide.practice.task) || !text(guide.practice.hint) || !text(guide.practice.solution)
                || !Array.isArray(guide.practice.cases) || guide.practice.cases.length < 2
                || !guide.practice.cases.every(test => sameKeys(test, ["input", "expected"]) && text(test.input) && text(test.expected))
                || !sameKeys(guide.interview, ["question", "answer"])
                || !text(guide.interview.question) || !text(guide.interview.answer) || !texts(guide.readiness)) {
                throw new Error(`Invalid OS/network learningGuide schema at ${label}`);
            }
            if (modules[index].number !== expectedNumbers[key][index]) {
                throw new Error(`OS/network learningGuide index mismatch at ${label}: expected ${expectedNumbers[key][index]}`);
            }
            modules[index].learningGuide = guide;
        });
    }
})();
