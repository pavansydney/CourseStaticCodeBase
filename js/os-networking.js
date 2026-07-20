// ============================================================
// Operating Systems + Networking curriculum (deep rewrite)
// Parser-safe labels and Mermaid-friendly diagrams.
// ============================================================

/* global courseData */

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
Worked Example: A text editor writes a file without knowing SSD firmware commands because the OS exposes a filesystem API.
Common Mistakes: Treating the OS as just a launcher instead of a safety and resource-management system.
Recap:
- OS provides abstraction and isolation
- Resource arbitration is continuous
- APIs decouple apps from hardware details
Practice:
- List three user actions that require kernel-managed resources`
            },
            {
                title: "Lesson 2: User Mode vs Kernel Mode",
                content: `Learning Objective: Explain why modern systems separate unprivileged and privileged execution.
Core Theory: User mode runs normal applications with restricted instruction access. Kernel mode executes trusted code with full hardware privilege. Controlled transitions protect system integrity by requiring explicit entry points and validation.
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
Worked Example: read(), write(), and socket() wrappers eventually invoke architecture-specific syscall instructions.
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
Worked Example: A web worker blocks on disk I/O, then returns to ready when the completion interrupt arrives.
Common Mistakes: Thinking blocked tasks still consume CPU time.
Recap:
- State transitions model readiness and progress
- PCB enables pause/resume correctness
- I/O completion drives unblock events
Practice:
- Identify two PCB fields required for context restore`
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
Worked Example: Desktop OS favors responsive UI tasks while background compile jobs receive lower dynamic priority.
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
Worked Example: Two processes can use the same virtual address value while mapping to different physical memory.
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
Core Theory: Page-based translation requires lookup on every memory reference. TLB caches recent mappings. TLB misses trigger page table walks, increasing access cost.
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
Core Theory: Demand paging loads pages lazily when first touched. Minor faults avoid storage I/O; major faults require disk access and are much slower.
Diagram (Mermaid):
flowchart LR
A[Access page] --> B{In RAM}
B -->|No| C[Page fault trap]
C --> D[Load from disk]
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
Core Theory: Data races occur when concurrent unsynchronized accesses include writes and correctness depends on timing. Synchronization constructs create ordering and visibility guarantees needed for deterministic behavior.
Diagram (Mermaid):
flowchart LR
A[Thread A write] --> B[Shared state]
C[Thread B read] --> B
B --> D{Synchronized}
D -->|No| E[Nondeterministic result]
Worked Example: Two worker threads increment same counter without atomicity, losing updates under load.
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
Worked Example: The same application writes to local SSD and network-mounted filesystem through unchanged calls.
Common Mistakes: Assuming identical durability semantics across all filesystems.
Recap:
- VFS decouples API from implementation
- Filesystem semantics differ in edge cases
- Portability requires explicit durability assumptions
Practice:
- Compare one semantic difference between two filesystems`
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
Core Theory: write() usually updates cached pages first. Background writeback flushes to storage later. fsync and related calls force durability for required records.
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
Worked Example: HTTPS traffic uses application semantics over transport reliability and routed IP delivery.
Common Mistakes: Memorizing layers without understanding failure boundaries.
Recap:
- Layer boundaries simplify protocol evolution
- Troubleshooting is faster when mapped by layer
- Protocols compose, not replace each other
Practice:
- Place DNS, TCP, and Ethernet into correct layers`
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
Worked Example: First packet to a local service triggers ARP request before unicast frame delivery.
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
Core Theory: Hosts consult local route table. Remote destinations are sent to default gateway. Each router selects next hop by destination prefix and forwarding policy.
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
Core Theory: Links enforce MTU limits. Oversized packets may fragment or be dropped depending on protocol/path behavior. Path MTU discovery helps endpoints adjust segment size.
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
Core Theory: Prefix length divides network and host bits. Larger prefix number means smaller subnet range. Effective host capacity differs from raw count due to reserved addresses.
Diagram (Mermaid):
flowchart LR
A[IP block] --> B[Prefix length]
B --> C[Network scope]
B --> D[Host range]
Worked Example: Splitting 10.0.0.0/24 into four /26 subnets for web, app, data, and operations tiers.
Common Mistakes: Miscounting usable addresses in cloud environments.
Recap:
- Prefix math drives allocation planning
- Subnet sizing affects growth and policy
- Reservation rules must be considered
Practice:
- Divide a /24 into eight equal subnets`
            },
            {
                title: "Lesson 2: Private and Public Address Boundaries",
                content: `Learning Objective: Explain where private addressing works and where public routing is required.
Core Theory: Private ranges support internal segmentation and are not internet-routable. Public addresses are needed at exposure points like load balancers, NAT gateways, or edge firewalls.
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
Worked Example: API gateway improves latency by reusing upstream keep-alive connections instead of reconnecting per request.
Common Mistakes: Ignoring connection setup overhead in high-QPS microservice paths.
Recap:
- Connection lifecycle has measurable cost
- Reuse can reduce repeated handshake latency
- State transitions help debug socket issues
Practice:
- Explain why connection pooling improves API performance`
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
Core Theory: UDP provides low-overhead datagram transport with no built-in ordering or retransmission. Applications requiring timeliness may accept loss and implement selective recovery, jitter buffering, or FEC.
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
Core Theory: QUIC runs over UDP and integrates secure handshake and stream multiplexing to reduce setup latency and head-of-line blocking effects across streams.
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
Core Theory: Resolution involves local cache, recursive resolver cache, and authoritative hierarchy when needed. TTL values trade freshness for query load.
Diagram (Mermaid):
flowchart LR
A[Client cache] --> B[Recursive resolver]
B --> C[Authoritative chain]
C --> B
B --> A
Worked Example: Record cutover appears partial because some resolvers still hold old cached value.
Common Mistakes: Debugging HTTP before validating name resolution.
Recap:
- DNS is distributed and cache-heavy
- TTL controls propagation timing
- Resolver behavior affects incident scope
Practice:
- Build a rollback checklist for incorrect DNS change`
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
