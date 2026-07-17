// ============================================================
// DSA + System Design curriculum for broad audiences:
// college grads, non-IT learners, and working engineers.
// Loaded on Courses page after script.js.
// ============================================================

/* global courseData */

courseData.dsaFoundations = [
    {
        number: "DSA · Module 1",
        title: "Problem Solving Mindset & Big-O",
        description: "Learn how to think through coding problems and estimate performance before writing code.",
        duration: "55 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Input-output thinking", "Brute force to optimized", "Time complexity", "Space complexity", "Dry runs and edge cases"],
        detailedDescription: "Great DSA performance starts with clear thinking, not syntax tricks. This module gives a universal problem-solving framework and teaches complexity analysis in plain language.",
        detailedContent: [
            {
                title: "A repeatable problem-solving framework",
                content: `Use this sequence for every problem:
1. Clarify inputs, outputs, constraints.
2. Solve with brute force first.
3. Identify bottlenecks.
4. Optimize with the right data structure or pattern.
5. Validate with edge cases and dry runs.

This works for beginners and experienced engineers alike.`
            },
            {
                title: "Big-O without fear",
                content: `Big-O helps compare scalability as input grows.

Common complexities:
• O(1) constant
• O(log n) binary search style
• O(n) single pass
• O(n log n) efficient sorting
• O(n^2) nested loops

Interviewers value your reasoning as much as your final code.`
            },
            {
                title: "From brute force to optimized",
                content: `Always show progression from naive to better.

Example progression:
• Check each pair: O(n^2)
• Use hash map: O(n)

This demonstrates engineering maturity and trade-off awareness.`,
                code: `def has_pair_with_sum(nums, target):
    seen = set()
    for x in nums:
        if target - x in seen:
            return True
        seen.add(x)
    return False`
            }
        ]
    },
    {
        number: "DSA · Module 2",
        title: "Core Data Structures from Scratch",
        description: "Arrays, strings, linked lists, stacks, queues, hash maps, sets, and heaps with practical use cases.",
        duration: "60 min",
        lessons: "6 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Arrays and strings", "Linked list basics", "Stack and queue", "Hashing", "Heap intuition", "Choosing the right structure"],
        detailedDescription: "Data structures are tools. This module helps learners choose the right one quickly based on operation costs and problem patterns.",
        detailedContent: [
            {
                title: "Choose by operations, not by habit",
                content: `Ask which operations dominate:
• Fast lookup: hash map/set
• Ordered retrieval: heap/tree
• LIFO workflows: stack
• FIFO workflows: queue
• Sequential processing: array/list

Right structure choice often converts hard problems into simple ones.`
            },
            {
                title: "Hash map pattern",
                content: `Hash maps solve counting, lookup, and frequency problems efficiently.

Typical use cases:
• duplicate checks
• anagram grouping
• frequency tracking
• first unique element`
            },
            {
                title: "Heap for top-k and streaming",
                content: `Heaps are best for dynamic min/max queries.

Use them for:
• top-k largest/smallest
• running median
• scheduling and priority systems`,
                code: `import heapq

def top_k_largest(nums, k):
    heap = []
    for n in nums:
        heapq.heappush(heap, n)
        if len(heap) > k:
            heapq.heappop(heap)
    return sorted(heap, reverse=True)`
            }
        ]
    },
    {
        number: "DSA · Module 3",
        title: "Sorting, Searching, and Two-Pointer Patterns",
        description: "Master universal interview patterns that solve many array and string problems quickly.",
        duration: "55 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Binary search", "Sort-based transformations", "Two pointers", "Sliding window", "Prefix sums"],
        detailedDescription: "This module builds pattern recognition - a top skill for coding interviews and day-to-day problem solving.",
        detailedContent: [
            {
                title: "Two pointers and sliding window",
                content: `These patterns reduce nested loops into linear scans.

Use two pointers for:
• sorted array pair problems
• in-place partitioning

Use sliding window for:
• longest/shortest subarray constraints
• substring problems with counts`
            },
            {
                title: "Binary search beyond arrays",
                content: `Binary search applies to any monotonic condition, not just sorted arrays.

Examples:
• minimum capacity problems
• feasibility checks
• optimization with yes/no predicate`
            },
            {
                title: "Sliding window example",
                content: `Track a moving window and update state incrementally.`,
                code: `def max_sum_subarray_k(nums, k):
    if k > len(nums):
        return None
    window = sum(nums[:k])
    best = window
    for i in range(k, len(nums)):
        window += nums[i] - nums[i - k]
        best = max(best, window)
    return best`
            }
        ]
    },
    {
        number: "DSA · Module 4",
        title: "Recursion, Backtracking, Trees, and Graph Basics",
        description: "Build confidence with recursion, DFS/BFS, binary trees, and graph traversal strategies.",
        duration: "65 min",
        lessons: "6 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Recursion mental model", "Backtracking", "Tree traversals", "Graph representation", "DFS vs BFS", "Visited state"],
        detailedDescription: "This module bridges beginner DSA to intermediate interview-level problem solving through tree and graph traversal foundations.",
        detailedContent: [
            {
                title: "Think in state and decision trees",
                content: `Recursion/backtracking is easier when you define:
• state at each call
• decision choices
• base case
• undo step (backtrack)

This model works for subsets, permutations, and path problems.`
            },
            {
                title: "DFS vs BFS",
                content: `DFS:
• stack/recursion
• good for path exploration and components

BFS:
• queue
• best for shortest path in unweighted graphs and level order problems`
            },
            {
                title: "Level-order traversal",
                content: `BFS naturally processes tree levels.`,
                code: `from collections import deque

def level_order(root):
    if not root:
        return []
    q = deque([root])
    out = []
    while q:
        node = q.popleft()
        out.append(node.val)
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    return out`
            }
        ]
    }
];

courseData.dsaInterview = [
    {
        number: "DSA · Module 5",
        title: "Dynamic Programming Without Memorization",
        description: "Understand DP through state transitions and overlapping subproblems, not formula memorization.",
        duration: "60 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["When DP applies", "Top-down memoization", "Bottom-up tabulation", "State transition design", "Space optimization"],
        detailedDescription: "Most learners fear DP because they memorize patterns blindly. This module teaches a practical method to derive DP from first principles.",
        detailedContent: [
            {
                title: "DP checklist",
                content: `Use DP when problems have:
• overlapping subproblems
• optimal substructure

Design steps:
1. Define state.
2. Write transition.
3. Set base case.
4. Choose memoization or tabulation.`
            },
            {
                title: "Top-down to bottom-up conversion",
                content: `Start with recursion + memo (easy to reason), then convert to iterative table for performance and control.`
            },
            {
                title: "Classic Fibonacci DP",
                content: `Simple example of recurrence and memoization.`,
                code: `def fib(n, memo=None):
    memo = memo or {}
    if n <= 1:
        return n
    if n in memo:
        return memo[n]
    memo[n] = fib(n - 1, memo) + fib(n - 2, memo)
    return memo[n]`
            }
        ]
    },
    {
        number: "DSA · Module 6",
        title: "Advanced Graphs, Greedy, and Union-Find",
        description: "Prepare for strong company-level interviews with advanced but teachable patterns.",
        duration: "60 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Topological sort", "Shortest paths", "Minimum spanning tree basics", "Greedy strategy", "Disjoint set union"],
        detailedDescription: "This module covers high-value topics that often separate average and strong interview performance.",
        detailedContent: [
            {
                title: "Topological sort and dependency resolution",
                content: `Use topological sort for ordering tasks with prerequisites (course schedule, build systems, workflows).`
            },
            {
                title: "Greedy correctness",
                content: `Greedy works only when local optimal choices guarantee global optimality. You must justify this in interviews.`
            },
            {
                title: "Union-Find skeleton",
                content: `Disjoint set union supports near O(1) union/find with path compression and union by rank.`,
                code: `class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True`
            }
        ]
    },
    {
        number: "DSA · Module 7",
        title: "Interview Strategy: Communication, Dry Run, and Trade-offs",
        description: "Learn how to perform in live coding interviews with clarity, confidence, and structured communication.",
        duration: "45 min",
        lessons: "4 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Clarifying questions", "Think aloud effectively", "Test case strategy", "Optimization discussion"],
        detailedDescription: "Strong candidates do not just solve problems - they communicate engineering decisions and trade-offs clearly.",
        detailedContent: [
            {
                title: "Interview communication flow",
                content: `Recommended flow:
1. Rephrase problem.
2. Confirm constraints.
3. Explain brute force.
4. Propose optimized approach.
5. Code with checkpoints.
6. Test edge cases.
7. Discuss complexity.`
            },
            {
                title: "Dry-run discipline",
                content: `Dry run your algorithm before coding and again with sample input after coding. This catches off-by-one and null-case bugs early.`
            },
            {
                title: "Complexity summary template",
                content: `Close answers with a concise summary:
• Time complexity: O(...)
• Space complexity: O(...)
• Why this is optimal (or near-optimal)`
            }
        ]
    },
    {
        number: "DSA · Module 8",
        title: "90-Day DSA Roadmap and Practice Engine",
        description: "A practical training plan for consistent daily progress from beginner to interview-ready.",
        duration: "40 min",
        lessons: "4 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Daily schedule", "Topic rotation", "Revision loops", "Mock interview cycles"],
        detailedDescription: "This module converts concepts into a working routine with weekly goals, revision cycles, and confidence tracking.",
        detailedContent: [
            {
                title: "Weekly cadence that works",
                content: `A practical weekly loop:
• 4 days new problems
• 2 days revision and pattern recap
• 1 day mock interview + reflection

Consistency beats bursts.`
            },
            {
                title: "Revision system",
                content: `Tag solved problems by pattern and difficulty. Revisit at 3-day, 7-day, and 21-day intervals to retain problem templates.`
            },
            {
                title: "Progress metrics",
                content: `Track:
• solve rate under time limit
• no-hint success ratio
• communication quality in mock sessions

These metrics predict interview readiness better than total problem count.`
            }
        ]
    }
];

courseData.systemDesignFoundations = [
    {
        number: "System Design · Module 1",
        title: "System Design Fundamentals for Everyone",
        description: "Understand scale, latency, throughput, and reliability in simple language with practical examples.",
        duration: "50 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["HLD vs LLD", "Functional vs non-functional requirements", "Scale vocabulary", "CAP basics", "Availability and SLOs", "Design trade-offs"],
        detailedDescription: "This module makes system design approachable for non-IT and IT learners by teaching first principles before architecture diagrams.",
        detailedContent: [
            {
                title: "HLD vs LLD (High-Level vs Low-Level Design)",
                content: `A common beginner doubt: what is the difference between <strong>HLD</strong> and <strong>LLD</strong>?

Simple analogy:
• <strong>HLD is the city map</strong> - shows the big picture and how areas connect.
• <strong>LLD is the building blueprint</strong> - shows exactly how one building is constructed.

<strong>High-Level Design (HLD):</strong>
• Focuses on overall system architecture
• Major components and how they interact
• Scalability, availability, reliability, and data flow
• Example (chat app): API gateway, chat service, message queue, database, cache, WebSocket layer

<strong>Low-Level Design (LLD):</strong>
• Focuses on implementation details inside a component
• Classes, interfaces, methods, and database schema
• Design patterns, validations, and error handling
• Example (chat service): MessageService class, sendMessage() method, message table schema, retry logic

<strong>Quick rule:</strong>
• HLD answers: "What are the parts and how do they scale?"
• LLD answers: "Exactly how is this part built in code and data?"

<strong>In interviews:</strong> System design rounds usually start with HLD, then the interviewer may ask for LLD of one component to test depth.`
            },
            {
                title: "Start with requirements",
                content: `Every design starts by separating:
• Functional requirements (what system does)
• Non-functional requirements (how well it does it)

Most interview mistakes come from skipping this step.`
            },
            {
                title: "Think in trade-offs",
                content: `You cannot maximize everything simultaneously.

Typical trade-offs:
• latency vs consistency
• cost vs performance
• simplicity vs flexibility`
            },
            {
                title: "Capacity estimation basics",
                content: `Rough numbers guide architecture choices.

Estimate:
• requests per second
• storage per day
• peak traffic multiplier`
            }
        ]
    },
    {
        number: "System Design · Module 2",
        title: "Building Blocks: API, DB, Cache, Queue, and CDN",
        description: "Learn when and why to use common infrastructure components in real systems.",
        duration: "60 min",
        lessons: "6 lessons",
        isNew: true,
        isLocked: false,
        topics: ["REST and idempotency", "SQL vs NoSQL", "Caching tiers", "Message queues", "CDN and edge", "Load balancing"],
        detailedDescription: "This module gives a practical map of modern backend building blocks and when each one is appropriate.",
        detailedContent: [
            {
                title: "SQL vs NoSQL",
                content: `Choose based on access patterns:
• SQL for relational integrity and complex joins
• NoSQL for flexible schema and horizontal scale

Hybrid architectures are common.`
            },
            {
                title: "Caching strategy",
                content: `Use cache to reduce latency and database load.

Patterns:
• cache-aside
• write-through
• write-back

Plan cache invalidation from day one.`
            },
            {
                title: "Queue decoupling",
                content: `Queues smooth traffic spikes and isolate slow workflows.

Good for:
• notifications
• media processing
• analytics pipelines`
            }
        ]
    },
    {
        number: "System Design · Module 3",
        title: "Scaling and Reliability Patterns",
        description: "Partitioning, replication, sharding, failover, and observability patterns for resilient systems.",
        duration: "60 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Horizontal scaling", "Replication", "Sharding", "Rate limiting", "Observability and alerts"],
        detailedDescription: "This module introduces production reliability patterns used by high-scale systems.",
        detailedContent: [
            {
                title: "Sharding and partitioning",
                content: `When a single database cannot handle load, partition data by key.

Challenges:
• hot partitions
• cross-shard queries
• rebalancing`
            },
            {
                title: "Rate limiting and backpressure",
                content: `Protect services with rate limits, quotas, and graceful degradation under peak demand.`
            },
            {
                title: "Observability stack",
                content: `Production confidence requires:
• logs for events
• metrics for trends
• traces for request paths
• alerts tied to SLOs`
            }
        ]
    },
    {
        number: "System Design · Module 4",
        title: "Design Documents and Architecture Communication",
        description: "Translate architecture ideas into clear documents and diagrams that teams can implement.",
        duration: "45 min",
        lessons: "4 lessons",
        isNew: true,
        isLocked: false,
        topics: ["High-level design template", "API contracts", "Data model sketching", "Risk and rollout planning"],
        detailedDescription: "Good architecture fails without clear communication. This module teaches design docs that align teams quickly.",
        detailedContent: [
            {
                title: "System design doc template",
                content: `Suggested sections:
1. Requirements
2. Capacity estimates
3. High-level architecture
4. Data model
5. API contracts
6. Bottlenecks and mitigations
7. Rollout strategy`
            },
            {
                title: "Design reviews",
                content: `Invite feedback early. Strong reviews focus on assumptions, failure modes, and operations, not only happy paths.`
            },
            {
                title: "Rollout and rollback",
                content: `Plan release safety with canaries, feature flags, and rollback triggers before going live.`
            }
        ]
    }
];

courseData.systemDesignAdvanced = [
    {
        number: "System Design · Module 5",
        title: "Case Study: URL Shortener",
        description: "Design a robust short-link platform with analytics, custom aliases, and high read traffic.",
        duration: "45 min",
        lessons: "4 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Key generation", "Redirect latency", "Read-heavy optimization", "Expiration and abuse handling"],
        detailedDescription: "A perfect starter case study for system design interviews: simple product, rich architecture trade-offs.",
        detailedContent: [
            {
                title: "Core architecture",
                content: `Typical design:
• API gateway + app service
• key generation service
• primary DB for mapping
• cache for hot links
• async analytics pipeline`
            },
            {
                title: "Hot path optimization",
                content: `Redirect endpoint must be low-latency. Cache short->long mappings aggressively and avoid unnecessary writes on read path.`
            },
            {
                title: "Abuse controls",
                content: `Add rate limiting, URL safety checks, and domain reputation controls to prevent spam and malicious links.`
            }
        ]
    },
    {
        number: "System Design · Module 6",
        title: "Case Study: Chat and Notification System",
        description: "Real-time messaging architecture with WebSockets, fan-out, delivery guarantees, and push notifications.",
        duration: "55 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["WebSocket gateway", "Message ordering", "Offline delivery", "Fan-out strategies", "Push notification pipeline"],
        detailedDescription: "Learn how modern chat systems handle real-time communication, persistence, and reliability at scale.",
        detailedContent: [
            {
                title: "Real-time gateway",
                content: `Use stateful WebSocket gateways for persistent connections and route messages through a pub/sub or queue backbone.`
            },
            {
                title: "Ordering and delivery",
                content: `Define guarantees clearly:
• at-most-once
• at-least-once
• exactly-once (expensive)

Most systems use idempotency and deduplication with at-least-once delivery.`
            },
            {
                title: "Offline and sync",
                content: `Store messages durably, track per-device cursors, and support history sync when users reconnect.`
            }
        ]
    },
    {
        number: "System Design · Module 7",
        title: "Case Study: Social Feed",
        description: "Design a scalable feed system with fan-out strategies, ranking, and timeline consistency.",
        duration: "55 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Fan-out on write vs read", "Ranking pipeline", "Cold start handling", "Caching and invalidation", "Consistency model"],
        detailedDescription: "Feed systems combine ranking, caching, and massive scale challenges - ideal for advanced interview prep.",
        detailedContent: [
            {
                title: "Fan-out strategies",
                content: `Fan-out on write gives low read latency but heavy write amplification.
Fan-out on read reduces write cost but increases read computation.

Hybrid approach is common.`
            },
            {
                title: "Ranking service",
                content: `Use a ranking layer to score content by relevance and freshness; update ranking features asynchronously.`
            },
            {
                title: "Cache strategy",
                content: `Multi-layer caching (edge + app + data cache) is essential. Define freshness windows and fallback behavior.`
            }
        ]
    },
    {
        number: "System Design · Module 8",
        title: "Case Study: Video Streaming Platform",
        description: "Design upload, transcoding, storage, CDN delivery, and playback telemetry for large-scale streaming.",
        duration: "60 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Upload pipeline", "Transcoding workflow", "Chunked delivery", "CDN strategy", "Playback analytics"],
        detailedDescription: "This capstone case study connects everything: async processing, object storage, global delivery, and observability.",
        detailedContent: [
            {
                title: "Media processing pipeline",
                content: `Standard flow:
1. Upload to object storage
2. Trigger transcoding jobs
3. Generate adaptive bitrate variants
4. Publish manifests
5. Serve via CDN`
            },
            {
                title: "Global delivery",
                content: `Use CDN edge caching and adaptive bitrate streaming to optimize startup time and reduce buffering.`
            },
            {
                title: "Quality telemetry",
                content: `Track QoE metrics:
• startup time
• rebuffer ratio
• bitrate switches
• completion rate

Use these to improve encoding and delivery policy.`
            }
        ]
    }
];
