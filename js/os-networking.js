// ============================================================
// Operating Systems + Networking curriculum for broad audiences:
// college grads, non-IT learners, and working engineers.
// Loaded on Courses page after script.js.
// ============================================================

/* global courseData */

// ---------- Track 1: OS Fundamentals ----------
courseData.osFundamentals = [
    {
        number: "OS · Module 1",
        title: "What an Operating System Really Does",
        description: "The role of the OS as the manager between your programs and the hardware.",
        duration: "45 min",
        lessons: "4 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Why we need an OS", "Kernel vs user space", "System calls", "OS responsibilities"],
        detailedDescription: "Before diving into processes and memory, understand the big picture: an operating system is a resource manager and a safety layer between applications and hardware.",
        detailedContent: [
            {
                title: "The OS as a resource manager",
                content: `An operating system sits between your applications and the hardware and manages shared resources so programs do not step on each other.

It manages:
• CPU time (which program runs now)
• Memory (who gets which memory)
• Storage (files and directories)
• Devices (keyboard, disk, network)

Without an OS, every program would have to control hardware directly and safely coordinate with every other program.`
            },
            {
                title: "Kernel space vs user space",
                content: `Modern systems split execution into two worlds:

<strong>Kernel space:</strong>
• The trusted core of the OS
• Full access to hardware
• Runs the scheduler, memory manager, drivers

<strong>User space:</strong>
• Where your apps run
• Limited, protected access
• Must ask the kernel for privileged operations

This separation protects the system: a buggy app cannot directly crash the whole machine.`
            },
            {
                title: "System calls: the bridge",
                content: `A <strong>system call</strong> is how a user program requests a service from the kernel (open a file, send data, create a process).

Everyday examples:
• read() / write() for files
• fork() / exec() to run programs
• socket() / send() for networking

When you call a high-level function like print, it eventually triggers a system call under the hood.`
            },
            {
                title: "Core OS responsibilities",
                content: `The OS is responsible for:
• Process management (running programs)
• Memory management (allocating RAM)
• File system management (organizing storage)
• Device/driver management (talking to hardware)
• Security and access control (permissions)

Every later module in this track expands one of these responsibilities.`
            }
        ]
    },
    {
        number: "OS · Module 2",
        title: "How Programs Run: From Code to Execution",
        description: "What happens between clicking 'run' and your program executing on the CPU.",
        duration: "45 min",
        lessons: "4 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Program vs process", "Compilation and loading", "The CPU fetch-execute cycle", "Interrupts"],
        detailedDescription: "This module demystifies execution: how a stored program becomes a running process, and how the CPU actually carries out instructions.",
        detailedContent: [
            {
                title: "Program vs process",
                content: `A <strong>program</strong> is a passive file on disk (instructions + data).
A <strong>process</strong> is a program that is actively running, with its own memory, registers, and state.

Analogy:
• Program = a recipe written on paper
• Process = actually cooking that recipe in the kitchen right now

You can run the same program multiple times, creating multiple independent processes.`
            },
            {
                title: "From source code to a running process",
                content: `Typical journey:
1. Write source code
2. Compile/interpret into machine-executable form
3. OS loads it into memory
4. OS creates a process and gives it CPU time
5. The program runs, using system calls for OS services

Interpreted languages (like Python) add a runtime that executes the code, but the OS still manages the process.`
            },
            {
                title: "The fetch-decode-execute cycle",
                content: `At its core, a CPU repeats a simple loop billions of times per second:

1. <strong>Fetch</strong> the next instruction from memory
2. <strong>Decode</strong> what it means
3. <strong>Execute</strong> it (math, memory access, jump)
4. Repeat

Everything your computer does is built on this tiny, relentless cycle.`
            },
            {
                title: "Interrupts: how the CPU multitasks",
                content: `An <strong>interrupt</strong> is a signal that tells the CPU to pause what it is doing and handle something urgent (a key press, network packet, or timer).

Why it matters:
• Enables responsiveness (react to devices instantly)
• Powers multitasking (timer interrupts let the OS switch programs)
• Avoids wasteful constant polling

Interrupts are how a single CPU appears to do many things at once.`
            }
        ]
    },
    {
        number: "OS · Module 3",
        title: "File Systems & Storage Basics",
        description: "How the OS organizes data on disk into files, directories, and permissions.",
        duration: "40 min",
        lessons: "4 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Files and directories", "File system structure", "Permissions", "Storage vs memory"],
        detailedDescription: "Storage is where your data lives permanently. This module explains how the OS turns raw disk blocks into an organized, secure file system.",
        detailedContent: [
            {
                title: "Files, directories, and paths",
                content: `A <strong>file</strong> is a named collection of data. A <strong>directory</strong> (folder) organizes files into a tree.

Key ideas:
• Absolute path: full location from the root (/home/user/report.txt)
• Relative path: location from where you are (./report.txt)
• Extensions hint at file type but do not guarantee it

The file system gives humans and programs a simple, hierarchical view over messy physical storage.`
            },
            {
                title: "How the file system maps to disk",
                content: `The OS hides physical complexity behind a clean abstraction.

Under the hood:
• Disks store data in fixed-size blocks
• Metadata (like inodes) tracks where a file's blocks live
• The file system maps file names to those blocks

You see report.txt; the OS knows exactly which disk blocks hold it.`
            },
            {
                title: "Permissions and access control",
                content: `File permissions decide who can read, write, or execute a file.

Common model (Unix-style):
• Owner, group, others
• Read (r), write (w), execute (x)

Permissions are a core security boundary: they stop one user or program from tampering with another's files.`
            },
            {
                title: "Storage vs memory (disk vs RAM)",
                content: `Two very different things people often confuse:

<strong>RAM (memory):</strong>
• Fast, temporary, lost on power off
• Holds running programs and active data

<strong>Disk (storage):</strong>
• Slower, permanent, survives power off
• Holds files and installed programs

The OS constantly moves data between disk and RAM as programs run.`
            }
        ]
    }
];

// ---------- Track 2: Processes, Memory & Concurrency ----------
courseData.osProcessesMemory = [
    {
        number: "OS · Module 4",
        title: "Processes & Threads",
        description: "How the OS represents running programs and enables multitasking with threads.",
        duration: "50 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Process states", "Process control block", "Threads vs processes", "Context switching", "Inter-process communication"],
        detailedDescription: "Processes and threads are the units of execution. This module explains how the OS tracks, switches, and coordinates them.",
        detailedContent: [
            {
                title: "The lifecycle of a process",
                content: `A process moves through states:
• <strong>New</strong> - being created
• <strong>Ready</strong> - waiting for CPU
• <strong>Running</strong> - executing now
• <strong>Waiting/Blocked</strong> - waiting for I/O or an event
• <strong>Terminated</strong> - finished

The OS scheduler constantly moves processes between Ready and Running.`
            },
            {
                title: "How the OS tracks a process",
                content: `Each process has a <strong>Process Control Block (PCB)</strong> - the OS's record card for it.

A PCB stores:
• Process ID
• Current state
• CPU registers (saved when paused)
• Memory info
• Open files

The PCB is what makes it possible to pause a process and resume it exactly where it left off.`
            },
            {
                title: "Threads vs processes",
                content: `A <strong>thread</strong> is a lightweight unit of execution inside a process.

Differences:
• Processes have separate memory; threads share their process's memory
• Threads are cheaper to create and switch
• A crash in one thread can affect the whole process

Threads let one application do several things at once (e.g., UI + background download).`
            },
            {
                title: "Context switching",
                content: `A <strong>context switch</strong> is when the CPU stops one process/thread and starts another.

Steps:
1. Save current process state (registers) into its PCB
2. Load the next process's state
3. Resume execution

Context switches enable multitasking but are not free - too many of them waste CPU time (overhead).`
            },
            {
                title: "Inter-process communication (IPC)",
                content: `Since processes have separate memory, they need explicit ways to talk.

Common IPC methods:
• Pipes
• Message queues
• Shared memory
• Sockets (also used across machines)

IPC is the foundation for how programs cooperate on a single machine.`
            }
        ]
    },
    {
        number: "OS · Module 5",
        title: "CPU Scheduling",
        description: "How the OS decides which process runs next and balances fairness with performance.",
        duration: "45 min",
        lessons: "4 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Why scheduling matters", "Scheduling goals", "Common algorithms", "Preemptive vs non-preemptive"],
        detailedDescription: "With many processes competing for a limited CPU, the scheduler decides who runs and for how long. This module covers the key strategies.",
        detailedContent: [
            {
                title: "Why scheduling exists",
                content: `There are usually more processes than CPU cores, so the OS must share CPU time.

Good scheduling delivers:
• Responsiveness (apps feel fast)
• Fairness (no process starves)
• High utilization (CPU stays busy)

Scheduling is a balancing act between competing goals.`
            },
            {
                title: "Scheduling goals and metrics",
                content: `Key metrics:
• <strong>Throughput</strong> - work completed per unit time
• <strong>Turnaround time</strong> - total time from arrival to completion
• <strong>Waiting time</strong> - time spent in the ready queue
• <strong>Response time</strong> - time until first response

Different systems prioritize different metrics (a server vs a phone).`
            },
            {
                title: "Common scheduling algorithms",
                content: `Classic algorithms to know:
• <strong>FCFS</strong> (First-Come, First-Served) - simple but can cause long waits
• <strong>SJF</strong> (Shortest Job First) - optimal average wait, needs job-length estimates
• <strong>Round Robin</strong> - each process gets a small time slice; great for responsiveness
• <strong>Priority scheduling</strong> - important tasks first (watch for starvation)

Real systems combine ideas (e.g., multilevel queues).`
            },
            {
                title: "Preemptive vs non-preemptive",
                content: `<strong>Non-preemptive:</strong> a running process keeps the CPU until it finishes or blocks.
<strong>Preemptive:</strong> the OS can forcibly pause a process (via timer interrupt) to run another.

Modern general-purpose OSes are preemptive - this is what keeps your system responsive even when one program is busy.`
            }
        ]
    },
    {
        number: "OS · Module 6",
        title: "Memory Management & Virtual Memory",
        description: "How the OS gives every process its own memory view using paging and virtual memory.",
        duration: "50 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Physical vs virtual memory", "Paging", "Address translation", "Page faults", "Swapping"],
        detailedDescription: "Memory management is one of the OS's most important jobs. This module explains virtual memory - the illusion that each process has its own large, private memory.",
        detailedContent: [
            {
                title: "The problem memory management solves",
                content: `Many processes must share limited physical RAM safely.

Challenges:
• Programs must not read/write each other's memory
• Programs may need more memory than physically exists
• Memory should be used efficiently

Virtual memory elegantly solves all three.`
            },
            {
                title: "Virtual vs physical memory",
                content: `Each process sees a private <strong>virtual address space</strong>, as if it owns a huge continuous block of memory.

The OS + hardware translate these virtual addresses to real <strong>physical addresses</strong> in RAM.

Benefits:
• Isolation (processes cannot see each other's memory)
• Simplicity (programs do not worry about physical layout)
• Flexibility (memory can be non-contiguous)`
            },
            {
                title: "Paging and address translation",
                content: `Memory is divided into fixed-size <strong>pages</strong> (virtual) and <strong>frames</strong> (physical).

A <strong>page table</strong> maps each virtual page to a physical frame. Hardware (the MMU) does this translation on every memory access, often accelerated by a cache called the TLB.

This is how scattered physical memory looks continuous to a program.`
            },
            {
                title: "Page faults",
                content: `A <strong>page fault</strong> happens when a program accesses a page that is not currently in RAM.

The OS then:
1. Pauses the process
2. Loads the needed page from disk
3. Updates the page table
4. Resumes the process

Occasional page faults are normal; too many (thrashing) severely slow the system.`
            },
            {
                title: "Swapping and thrashing",
                content: `When RAM is full, the OS moves less-used pages to disk (<strong>swap</strong>) to free space.

<strong>Thrashing</strong> occurs when the system spends more time swapping pages than doing real work - a sign there is not enough RAM for the current workload.

Fix: reduce active workload or add memory.`
            }
        ]
    },
    {
        number: "OS · Module 7",
        title: "Concurrency, Synchronization & Deadlocks",
        description: "How to coordinate multiple threads safely and avoid classic concurrency bugs.",
        duration: "50 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Race conditions", "Critical sections", "Locks and mutexes", "Semaphores", "Deadlocks"],
        detailedDescription: "Concurrency makes systems fast but introduces subtle bugs. This module teaches how to share data safely across threads and how deadlocks arise.",
        detailedContent: [
            {
                title: "Race conditions",
                content: `A <strong>race condition</strong> occurs when the result depends on the unpredictable timing of concurrent operations on shared data.

Classic example:
Two threads increment the same counter at once and one update is lost.

Race conditions are dangerous because they are intermittent and hard to reproduce.`
            },
            {
                title: "Critical sections",
                content: `A <strong>critical section</strong> is code that accesses shared data and must not run concurrently with itself.

Rule: only one thread should be inside the critical section at a time.

The goal of synchronization tools is to safely enforce this rule.`
            },
            {
                title: "Locks and mutexes",
                content: `A <strong>mutex</strong> (mutual exclusion lock) lets only one thread hold it at a time.

Pattern:
1. Acquire lock
2. Do critical work
3. Release lock

Warning: holding locks too long hurts performance; forgetting to release causes hangs.`
            },
            {
                title: "Semaphores",
                content: `A <strong>semaphore</strong> is a counter used to control access to a limited number of resources.

Uses:
• Binary semaphore (0/1) behaves like a lock
• Counting semaphore allows N concurrent users (e.g., 5 DB connections)

Semaphores are also used to signal between threads.`
            },
            {
                title: "Deadlocks",
                content: `A <strong>deadlock</strong> is when threads wait on each other forever.

Four conditions must all hold:
• Mutual exclusion
• Hold and wait
• No preemption
• Circular wait

Break any one condition to prevent deadlock (e.g., always acquire locks in a consistent order).`
            }
        ]
    }
];

// ---------- Track 3: Networking Fundamentals ----------
courseData.networkingFundamentals = [
    {
        number: "Networking · Module 1",
        title: "How Networks Work: The Big Picture",
        description: "What a network is and how the layered model organizes communication.",
        duration: "45 min",
        lessons: "4 lessons",
        isNew: true,
        isLocked: false,
        topics: ["What is a network", "OSI and TCP/IP models", "Encapsulation", "Clients and servers"],
        detailedDescription: "Networking can feel abstract. This module builds a clear mental model using the layered approach that all real networks follow.",
        detailedContent: [
            {
                title: "What a network actually is",
                content: `A <strong>network</strong> is two or more devices connected so they can exchange data.

Scales:
• LAN (local, like your home/office)
• WAN (wide area, connecting cities/countries)
• The Internet (a global network of networks)

At every scale, the same core ideas apply.`
            },
            {
                title: "The layered model (OSI & TCP/IP)",
                content: `Networking is organized into <strong>layers</strong>, each with one job.

TCP/IP (practical) layers:
• Application (HTTP, DNS)
• Transport (TCP, UDP)
• Internet (IP)
• Link (Ethernet, Wi-Fi)

The classic OSI model has 7 layers, but the 4-layer TCP/IP model maps to the real internet. Layering lets each part evolve independently.`
            },
            {
                title: "Encapsulation: data in envelopes",
                content: `As data goes down the layers, each layer wraps it with its own header - like putting a letter inside nested envelopes.

Flow (sending):
Application data -> add transport header -> add IP header -> add link header -> bits on the wire.

The receiver unwraps each layer in reverse. This is called <strong>encapsulation/decapsulation</strong>.`
            },
            {
                title: "Clients, servers, and ports",
                content: `Most communication uses the <strong>client-server</strong> model:
• Client requests (your browser)
• Server responds (a website)

A <strong>port number</strong> identifies which service on a machine (e.g., 80 for HTTP, 443 for HTTPS). An IP finds the machine; the port finds the app on it.`
            }
        ]
    },
    {
        number: "Networking · Module 2",
        title: "IP Addressing & Routing",
        description: "How devices are identified and how data finds its way across the internet.",
        duration: "45 min",
        lessons: "4 lessons",
        isNew: true,
        isLocked: false,
        topics: ["IPv4 and IPv6", "Public vs private IPs", "Subnets basics", "Routing and NAT"],
        detailedDescription: "IP addresses are the postal system of the internet. This module explains addressing, subnets, and how routers move packets toward their destination.",
        detailedContent: [
            {
                title: "IP addresses (IPv4 & IPv6)",
                content: `An <strong>IP address</strong> uniquely identifies a device on a network.

• <strong>IPv4</strong>: e.g., 192.168.1.10 (about 4 billion addresses - now scarce)
• <strong>IPv6</strong>: much larger address space to support the growing internet

Think of an IP as a mailing address for data.`
            },
            {
                title: "Public vs private IPs",
                content: `<strong>Private IPs</strong> are used inside local networks (home/office) and are not directly reachable from the internet (e.g., 192.168.x.x, 10.x.x.x).

<strong>Public IPs</strong> are globally routable on the internet.

Your devices share one public IP via your router using NAT (next lesson).`
            },
            {
                title: "Subnets in plain language",
                content: `A <strong>subnet</strong> splits a large network into smaller logical groups.

Why it helps:
• Organizes devices
• Improves security (isolation)
• Reduces unnecessary traffic

The subnet mask (like /24) marks which part of an IP is the network vs the host.`
            },
            {
                title: "Routing and NAT",
                content: `<strong>Routing</strong> is how packets hop across routers to reach a destination network, each router choosing the next best step.

<strong>NAT</strong> (Network Address Translation) lets many private devices share one public IP - your home router does this so all your devices can browse using a single public address.`
            }
        ]
    }
];

// ---------- Track 4: Web, DNS & Protocols ----------
courseData.networkingProtocols = [
    {
        number: "Networking · Module 3",
        title: "TCP vs UDP: Reliable vs Fast",
        description: "The two core transport protocols and when each is the right choice.",
        duration: "45 min",
        lessons: "4 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Transport layer role", "TCP reliability", "UDP speed", "Choosing TCP vs UDP"],
        detailedDescription: "TCP and UDP are the two ways applications send data across the internet. This module makes the trade-offs crystal clear.",
        detailedContent: [
            {
                title: "What the transport layer does",
                content: `The transport layer delivers data between applications on two machines, adding services on top of raw IP.

It handles:
• Identifying the app via ports
• Optionally ensuring reliability and ordering

Two dominant protocols: <strong>TCP</strong> and <strong>UDP</strong>.`
            },
            {
                title: "TCP: reliable and ordered",
                content: `<strong>TCP</strong> guarantees data arrives correctly and in order.

Features:
• Connection setup via the 3-way handshake (SYN, SYN-ACK, ACK)
• Acknowledgments and retransmission of lost data
• Flow and congestion control

Used by: web (HTTP/HTTPS), email, file transfer - anywhere correctness matters.`
            },
            {
                title: "UDP: fast and lightweight",
                content: `<strong>UDP</strong> sends data without connection setup or delivery guarantees.

Traits:
• No handshake, no retransmission
• Lower latency, less overhead
• Packets may be lost or reordered

Used by: video calls, live streaming, online gaming, DNS - where speed beats perfect reliability.`
            },
            {
                title: "Choosing TCP vs UDP",
                content: `Decision guide:
• Need every byte, in order? -> <strong>TCP</strong>
• Need speed and can tolerate some loss? -> <strong>UDP</strong>

Example:
• Loading a webpage -> TCP
• A live voice call -> UDP (a tiny glitch is better than a long delay)`
            }
        ]
    },
    {
        number: "Networking · Module 4",
        title: "DNS: The Internet's Address Book",
        description: "How human-friendly domain names get translated into IP addresses.",
        duration: "40 min",
        lessons: "4 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Why DNS exists", "DNS resolution steps", "Record types", "Caching and TTL"],
        detailedDescription: "DNS quietly powers almost every internet action. This module explains how a name like example.com becomes an IP address.",
        detailedContent: [
            {
                title: "Why DNS exists",
                content: `Humans remember names; computers route by numbers.

<strong>DNS</strong> (Domain Name System) translates domain names (example.com) into IP addresses (like 93.184.x.x).

Without DNS, you would have to memorize IP addresses for every website.`
            },
            {
                title: "How a DNS lookup works",
                content: `Simplified resolution steps:
1. Browser checks its cache
2. Ask the resolver (often your ISP)
3. Resolver queries the root server
4. Then the top-level domain (.com) server
5. Then the domain's authoritative server
6. IP is returned and cached

All of this usually happens in milliseconds.`
            },
            {
                title: "Common DNS record types",
                content: `Key records to know:
• <strong>A</strong> - maps a name to an IPv4 address
• <strong>AAAA</strong> - maps to an IPv6 address
• <strong>CNAME</strong> - alias one name to another
• <strong>MX</strong> - mail server for the domain
• <strong>TXT</strong> - text data (verification, security)

These records tell the internet how to reach your services.`
            },
            {
                title: "Caching and TTL",
                content: `DNS answers are cached to speed things up and reduce load.

<strong>TTL</strong> (Time To Live) says how long a record may be cached.

Trade-off:
• High TTL = faster, fewer lookups, slower to update
• Low TTL = quicker changes, more lookups

This is why DNS changes can take time to propagate.`
            }
        ]
    },
    {
        number: "Networking · Module 5",
        title: "HTTP, HTTPS & What Happens When You Visit a Website",
        description: "The web's core protocol and the full journey of a page request.",
        duration: "50 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["HTTP basics", "Methods and status codes", "HTTPS and TLS", "Request/response lifecycle", "End-to-end page load"],
        detailedDescription: "This capstone module ties OS and networking together by tracing exactly what happens from typing a URL to seeing a page.",
        detailedContent: [
            {
                title: "HTTP: the language of the web",
                content: `<strong>HTTP</strong> (HyperText Transfer Protocol) is how browsers and servers exchange web content.

Traits:
• Request-response based
• Stateless by default (each request is independent)
• Runs on top of TCP (and TLS for HTTPS)

Every page, image, and API call typically travels over HTTP(S).`
            },
            {
                title: "Methods and status codes",
                content: `<strong>Methods</strong> describe the action:
• GET (read), POST (create), PUT (update), DELETE (remove)

<strong>Status codes</strong> describe the result:
• 2xx success (200 OK)
• 3xx redirect (301/302)
• 4xx client error (404 Not Found)
• 5xx server error (500)

Reading status codes is a core debugging skill.`
            },
            {
                title: "HTTPS and TLS",
                content: `<strong>HTTPS</strong> is HTTP secured with <strong>TLS</strong> encryption.

TLS provides:
• Encryption (privacy)
• Integrity (data not tampered)
• Authentication (you are talking to the real site, via certificates)

The padlock in your browser means the connection is encrypted with TLS.`
            },
            {
                title: "The request/response lifecycle",
                content: `A single HTTP request typically involves:
1. DNS lookup to find the server IP
2. TCP connection (and TLS handshake for HTTPS)
3. Browser sends an HTTP request
4. Server processes and responds
5. Browser renders the response

Understanding this flow makes web debugging far easier.`
            },
            {
                title: "End-to-end: typing a URL to seeing a page",
                content: `Putting it all together when you visit a site:
1. Browser resolves the domain via <strong>DNS</strong>
2. Establishes a <strong>TCP</strong> connection (+ <strong>TLS</strong> for HTTPS)
3. Sends an <strong>HTTP</strong> request
4. Server (a running <strong>process</strong> managed by an <strong>OS</strong>) handles it
5. Response travels back across the <strong>network</strong>
6. Browser renders HTML, CSS, and JS

This single action uses processes, memory, TCP/IP, DNS, and HTTP together - everything in this track.`
            }
        ]
    }
];
