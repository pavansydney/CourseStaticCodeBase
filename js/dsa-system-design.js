// ============================================================
// DSA + System Design curriculum for broad audiences:
// college grads, non-IT learners, and working engineers.
// Loaded on Courses page after script.js.
// ============================================================

/* global courseData */

courseData.dsaFoundations = [
    {
        number: "DSA · Module 1",
        title: "Problem Solving Mindset, Big-O, and Correctness",
        description: "Build a repeatable approach for solving problems and explaining complexity with confidence.",
        duration: "90 min",
        lessons: "8 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Problem parsing", "Constraints to complexity", "Brute force baseline", "Invariants", "Dry runs", "Edge cases", "Optimization playbook", "Complexity communication"],
        detailedDescription: "This module turns problem solving into a disciplined workflow used by strong candidates in interviews and real engineering tasks.",
        detailedContent: [
            {
                title: "Lesson 0: DSA Introduction and How Interviews Are Evaluated",
                content: `Why this matters: coding rounds judge thinking quality, not just final syntax.
Learning Objective: understand the evaluation rubric used in most interviews.
Core Theory: interviewers assess clarity, progression from baseline to optimized, correctness reasoning, and testing discipline.
Diagram (Mermaid):
flowchart LR
A["Understand"] --> B["Baseline"]
B --> C["Optimize"]
C --> D["Code and test"]
Worked Example: describe pair-sum baseline and optimized approach in under one minute.
Common Mistakes: coding immediately without clarifying constraints.
Recap:
- think before coding
- explain progression clearly
- reserve time for testing
Practice:
- explain your approach flow for longest substring without repeating characters`
            },
            {
                title: "Lesson 1: Constraint-Driven Thinking",
                content: `Why this matters: constraints usually reveal the intended algorithm class.
Learning Objective: derive feasible complexity targets from input bounds.
Core Theory: if n is large, quadratic solutions are usually invalid; if data is sorted, two pointers or binary search may apply.
Diagram (Mermaid):
flowchart TD
A["Read constraints"] --> B["Set complexity target"]
B --> C["Choose candidate patterns"]
Worked Example: n up to 100000 suggests O(n) or O(n log n), rarely O(n^2).
Common Mistakes: ignoring duplicate and empty-input behavior.
Recap:
- constraints are first-class signals
- complexity target narrows design space
- edge cases should be listed early
Practice:
- pick feasible complexity for n=200000 and justify`
            },
            {
                title: "Lesson 2: Big-O, Theta, and Practical Cost",
                content: `Why this matters: growth-rate thinking avoids misleading micro-optimizations.
Learning Objective: compare algorithm scalability with precision.
Core Theory: Big-O is upper bound, Theta is tight bound, and constants matter mostly for small n.
Diagram (Mermaid):
flowchart LR
A["Input size"] --> B["Growth class"]
B --> C["Runtime at scale"]
Worked Example: O(n log n) sort outscales O(n^2) for moderate to large n.
Common Mistakes: treating O(2n) as different from O(n).
Recap:
- prioritize growth class first
- constants matter second
- always state assumptions
Practice:
- rank O(1), O(log n), O(n), O(n log n), O(n^2)`
            },
            {
                title: "Lesson 3: Invariants and Correctness Proof Basics",
                content: `Why this matters: correctness confidence comes from invariants, not guesswork.
Learning Objective: define and maintain loop invariants.
Core Theory: an invariant must hold before and after each iteration and lead to correctness at termination.
Diagram (Mermaid):
flowchart TD
A["Initialize invariant"] --> B["Maintain each iteration"]
B --> C["Terminate and conclude"]
Worked Example: binary search invariant for low and high bounds.
Common Mistakes: proving examples only, not the general case.
Recap:
- invariants build proof skeleton
- proof and complexity are separate checks
- dry run validates implementation details
Practice:
- write an invariant for two-pointer pair search`
            },
            {
                title: "Lesson 4: Baseline-to-Optimized Playbook",
                content: `Why this matters: interviewer trust increases when optimization is systematic.
Learning Objective: identify bottlenecks and apply targeted improvements.
Core Theory: optimize by removing repeated work, narrowing search space, or using better data structures.
Diagram (Mermaid):
flowchart LR
A["Brute force"] --> B["Identify bottleneck"]
B --> C["Apply pattern"]
C --> D["Recalculate complexity"]
Worked Example: duplicate detection from O(n^2) scan to O(n) hash-set pass.
Common Mistakes: changing multiple dimensions at once and losing correctness.
Recap:
- optimize one bottleneck at a time
- verify correctness after each change
- recalculate time and space explicitly
Practice:
- optimize an O(n^2) duplicate check problem`
            },
            {
                title: "Lesson 5: Interview Communication Template",
                content: `Why this matters: concise communication can be a decisive differentiator.
Learning Objective: deliver a complete solution summary in under 30 seconds.
Core Theory: include approach, complexity, trade-offs, and key edge-case behavior.
Worked Example: summarize hash-map two-sum with time O(n) and space O(n).
Common Mistakes: saying optimal without justification.
Recap:
- include both time and space
- mention assumptions and constraints
- explain why alternatives were not chosen
Practice:
- present one solved problem summary aloud`
            },
            {
                title: "Lesson 6: One-Pass State Tracking Example",
                content: `Why this matters: one-pass patterns appear across many categories.
Learning Objective: implement a clean one-pass complement lookup.
Core Theory: tracking visited values in a set enables O(1) average membership checks.
Diagram (Mermaid):
flowchart LR
A["Read value"] --> B["Check complement"]
B --> C{found}
C -- yes --> D["Return"]
C -- no --> E["Store value"]
Worked Example: pair sum existence with a visited set.
Common Mistakes: storing before checking can break index constraints in variant problems.
Recap:
- state order can affect correctness
- one-pass is often enough
- keep logic minimal and testable
Practice:
- return indices instead of boolean`,
                code: `def has_pair_with_sum(nums, target):
    seen = set()
    for x in nums:
        if target - x in seen:
            return True
        seen.add(x)
    return False`
            },
            {
                title: "Lesson 7: Recap and Readiness Gate",
                content: `Why this matters: retention requires retrieval practice, not passive rereading.
Learning Objective: verify you can solve and explain under time pressure.
Core Theory: readiness means you can state constraints, choose pattern, prove correctness, and test quickly.
Recap:
- constrain-first thinking
- baseline then optimize
- invariant-driven correctness
Practice:
- run one 25-minute mock with full explanation`
            }
        ]
    },
    {
        number: "DSA · Module 2",
        title: "Arrays, Strings, Hashing, and Prefix Patterns",
        description: "Deep mastery of the most frequently tested DSA patterns for linear data.",
        duration: "105 min",
        lessons: "8 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Frequency maps", "Prefix sums", "Sliding window", "Two pointers", "Difference arrays", "String scans", "Range queries", "Edge-case handling"],
        detailedDescription: "This module goes beyond basics and builds robust pattern-level skill for arrays and strings.",
        detailedContent: [
            {
                title: "Lesson 0: Arrays and Strings in Real Engineering",
                content: `Why this matters: array and string bugs are common in production and interviews.
Learning Objective: choose operations that match performance requirements.
Core Theory: arrays provide O(1) indexing, while string mutation cost depends on language representation.
Diagram (Mermaid):
flowchart LR
A["Contiguous memory"] --> B["Fast indexing"]
B --> C["Linear scans"]
Worked Example: avoiding repeated string concatenation in loops.
Common Mistakes: hidden quadratic behavior from repeated immutable string rebuilds.
Recap:
- choose operations with known complexity
- avoid hidden copies in tight loops
- test index boundaries aggressively
Practice:
- optimize a naive string-builder routine`
            },
            {
                title: "Lesson 1: Frequency Counting and Membership",
                content: `Why this matters: hash maps solve a large class of interview tasks quickly.
Learning Objective: apply count maps for duplicates, anagrams, and frequency ranking.
Core Theory: count dictionaries transform repeated scanning into incremental updates.
Diagram (Mermaid):
flowchart TD
A["Read element"] --> B["Update count"]
B --> C["Evaluate condition"]
Worked Example: first unique character with count map and second pass.
Common Mistakes: forgetting to initialize missing keys safely.
Recap:
- count map is a foundational pattern
- two-pass can still be optimal
- map vs set depends on multiplicity
Practice:
- solve valid anagram and explain Unicode caveats`
            },
            {
                title: "Lesson 2: Prefix Sum for Fast Range Queries",
                content: `Why this matters: repeated range queries need precomputation.
Learning Objective: build and query prefix arrays with correct boundaries.
Core Theory: range sum from l to r equals prefix r plus one minus prefix l.
Diagram (Mermaid):
flowchart LR
A["Build prefix once"] --> B["Answer many queries in O(1)"]
Worked Example: immutable range sum API.
Common Mistakes: off-by-one at left boundary zero.
Recap:
- add sentinel zero prefix
- write formula before coding
- prefer precompute for many queries
Practice:
- implement immutable range sum class`,
                code: `class NumArray:
    def __init__(self, nums):
        self.prefix = [0]
        for n in nums:
            self.prefix.append(self.prefix[-1] + n)

    def sum_range(self, left, right):
        return self.prefix[right + 1] - self.prefix[left]`
            },
            {
                title: "Lesson 3: Sliding Window Fixed and Variable",
                content: `Why this matters: sliding windows convert nested loops to linear scans.
Learning Objective: maintain valid window invariants while moving pointers.
Core Theory: fixed window updates by add-right remove-left; variable window grows and shrinks by constraint.
Diagram (Mermaid):
flowchart TD
A["Expand right"] --> B{window valid}
B -- no --> C["Shrink left"]
B -- yes --> D["Record best"]
D --> A
Worked Example: longest substring without repeating characters.
Common Mistakes: not updating state before moving left pointer.
Recap:
- define one invariant and enforce it
- pointer movement should be monotonic
- update answer only when invariant holds
Practice:
- solve minimum size subarray sum`
            },
            {
                title: "Lesson 4: Two-Pointer on Sorted Data",
                content: `Why this matters: sorted constraints can remove an entire loop.
Learning Objective: apply opposite-direction pointers and duplicate control.
Core Theory: when sum is too small move left, when too large move right.
Diagram (Mermaid):
flowchart LR
A["left and right"] --> B{sum compare target}
B -- small --> C["left plus one"]
B -- large --> D["right minus one"]
Worked Example: two-sum sorted and three-sum with duplicate skipping.
Common Mistakes: skipping duplicate handling in combination problems.
Recap:
- sorted inputs unlock linear scans
- direction of pointer movement follows comparator
- duplicate control affects correctness
Practice:
- implement 3-sum with no duplicate triplets`
            },
            {
                title: "Lesson 5: Difference Array for Batch Updates",
                content: `Why this matters: interval updates appear in scheduling and analytics.
Learning Objective: process many range updates in near linear time.
Core Theory: store delta at start and negative delta at end plus one, then prefix accumulate.
Diagram (Mermaid):
flowchart LR
A["Mark range deltas"] --> B["Prefix accumulation"]
B --> C["Final values"]
Worked Example: meeting room occupancy timeline.
Common Mistakes: forgetting end plus one boundary checks.
Recap:
- difference array is update-first optimization
- final prefix converts events to state
- useful when updates outnumber direct queries
Practice:
- solve car pooling capacity validation`
            },
            {
                title: "Lesson 6: Prefix plus Hashing Capstone",
                content: `Why this matters: combined patterns appear in medium-level interviews.
Learning Objective: count subarrays with target sum in O(n).
Core Theory: running prefix plus frequency map of previous prefixes tracks valid subarray counts.
Diagram (Mermaid):
flowchart LR
A["Update running sum"] --> B["Lookup sum minus target"]
B --> C["Add count"]
C --> D["Store running sum"]
Worked Example: subarray sum equals k.
Common Mistakes: using set instead of count map for multiplicity.
Recap:
- multiplicity requires map counts
- initialize prefix zero frequency one
- this pattern generalizes widely
Practice:
- adapt to longest subarray sum equals k`,
                code: `def subarray_sum(nums, k):
    counts = {0: 1}
    prefix = 0
    ans = 0
    for x in nums:
        prefix += x
        ans += counts.get(prefix - k, 0)
        counts[prefix] = counts.get(prefix, 0) + 1
    return ans`
            },
            {
                title: "Lesson 7: Recap and Drill Pack",
                content: `Why this matters: arrays and strings dominate early and mid interview rounds.
Learning Objective: quickly map prompts to hashing, prefix, window, or pointers.
Recap:
- hashing for membership and frequencies
- prefix for range and cumulative relationships
- window and pointers for linear constraints
Practice:
- complete one problem from each pattern family`
            }
        ]
    },
    {
        number: "DSA · Module 3",
        title: "Linked Lists, Stacks, Queues, and Deques",
        description: "Pointer-safe implementation and high-value ADT patterns used in interviews and systems code.",
        duration: "100 min",
        lessons: "7 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Linked list rewiring", "Fast and slow pointers", "Stack templates", "Queue workflows", "Deque and monotonic queue", "Sentinel nodes", "Common pitfalls"],
        detailedDescription: "This module trains pointer discipline and queue/stack pattern fluency required for robust implementations.",
        detailedContent: [
            {
                title: "Lesson 0: Pointer Structures and Operation Trade-offs",
                content: `Why this matters: structure choice controls performance and bug surface.
Learning Objective: choose list, stack, queue, or deque by operation profile.
Core Theory: linked structures optimize insert and delete near known nodes; arrays optimize random access.
Diagram (Mermaid):
flowchart LR
A["Operation profile"] --> B["Structure choice"]
B --> C["Complexity outcome"]
Worked Example: queue for task scheduling, stack for parser state.
Common Mistakes: choosing linked list for random indexing workloads.
Recap:
- operation profile first
- pointer updates require careful ordering
- sentinel nodes reduce branch complexity
Practice:
- choose structure for five workload descriptions`
            },
            {
                title: "Lesson 1: Reversing a Linked List Safely",
                content: `Why this matters: pointer rewiring tests mutation correctness discipline.
Learning Objective: reverse singly linked list with invariant tracking.
Core Theory: preserve next pointer before rewiring current node to previous.
Diagram (Mermaid):
flowchart TD
A["store next"] --> B["reverse current link"]
B --> C["advance pointers"]
Worked Example: iterative reverse in O(n) time and O(1) space.
Common Mistakes: losing remainder of list by rewiring too early.
Recap:
- keep next pointer safe
- invariant prev is reversed prefix head
- final prev is new head
Practice:
- reverse nodes in k-group`,
                code: `def reverse_list(head):
    prev = None
    curr = head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev`
            },
            {
                title: "Lesson 2: Fast and Slow Pointers",
                content: `Why this matters: cycle detection and midpoint tasks rely on dual-speed traversal.
Learning Objective: apply tortoise-hare method for cycle and mid problems.
Core Theory: fast moves two steps, slow moves one step; meeting implies cycle.
Diagram (Mermaid):
flowchart LR
A["slow plus one"] --> C{meet}
B["fast plus two"] --> C
C -- yes --> D["cycle found"]
Worked Example: detect cycle and explain why pointers meet.
Common Mistakes: null checks in wrong order for fast pointer.
Recap:
- two-speed traversal is O(1) extra space
- midpoint policy differs for even length
- cycle entry can be derived after meeting
Practice:
- check if linked list is palindrome`
            },
            {
                title: "Lesson 3: Stack Patterns",
                content: `Why this matters: stack models nested structures and deferred operations.
Learning Objective: solve parenthesis validation and monotonic stack basics.
Core Theory: last-in first-out behavior aligns with nested opening and closing tokens.
Diagram (Mermaid):
flowchart TD
A["read token"] --> B{opening token}
B -- yes --> C["push"]
B -- no --> D["validate with top"]
Worked Example: valid parentheses using match map.
Common Mistakes: forgetting final empty-stack check.
Recap:
- stack depth mirrors nesting depth
- validate each close token immediately
- final stack state matters
Practice:
- evaluate reverse polish notation`
            },
            {
                title: "Lesson 4: Queue, Deque, and Stream Processing",
                content: `Why this matters: real-time systems often process data streams incrementally.
Learning Objective: use deque for O(1) front operations and window summaries.
Core Theory: queue is FIFO; deque supports both ends efficiently.
Diagram (Mermaid):
flowchart LR
A["enqueue"] --> B["dequeue"]
C["appendleft"] --> D["popleft"]
Worked Example: moving average over stream with fixed window.
Common Mistakes: list pop from front causing O(n) degradation.
Recap:
- deque is the standard queue implementation in Python
- stream algorithms should avoid full recomputation
- window bounds must be explicit
Practice:
- implement sliding window maximum skeleton`,
                code: `from collections import deque

class MovingAverage:
    def __init__(self, size):
        self.size = size
        self.q = deque()
        self.total = 0

    def next(self, val):
        self.q.append(val)
        self.total += val
        if len(self.q) > self.size:
            self.total -= self.q.popleft()
        return self.total / len(self.q)`
            },
            {
                title: "Lesson 5: Sentinel Nodes and Bug Prevention",
                content: `Why this matters: edge-case branches are a common source of pointer bugs.
Learning Objective: simplify linked operations using sentinels and guard clauses.
Core Theory: dummy head and tail remove special handling for empty and single-node transitions.
Worked Example: doubly linked list insert and delete with sentinels.
Common Mistakes: not clearing detached node references in mutable structures.
Recap:
- sentinels reduce branch count
- pointer updates should be symmetric
- test empty and one-element cases first
Practice:
- design LRU cache structure with map plus doubly linked list`
            },
            {
                title: "Lesson 6: Recap and Pattern Drill",
                content: `Why this matters: stack and queue patterns combine heavily with tree and graph modules.
Learning Objective: map prompts to linked, stack, queue, or deque solutions quickly.
Recap:
- linked rewiring for mutation tasks
- stack for nested and monotonic behavior
- queue and deque for level and stream workflows
Practice:
- classify ten prompts by primary ADT before coding`
            }
        ]
    },
    {
        number: "DSA · Module 4",
        title: "Recursion, Backtracking, Trees, and Graph Fundamentals",
        description: "Bridge beginner and intermediate DSA with traversal depth, search space control, and graph modeling.",
        duration: "115 min",
        lessons: "8 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Recursion state model", "Backtracking template", "Tree DFS and BFS", "BST basics", "Graph representation", "Visited-state discipline", "Component search", "Traversal complexity"],
        detailedDescription: "This module introduces recursion and traversal as state-space exploration, building confidence for trees and graph interview questions.",
        detailedContent: [
            {
                title: "Lesson 0: Recursion as State Transition",
                content: `Why this matters: recursion becomes easy when state is explicit.
Learning Objective: define state, decisions, base case, and return value.
Core Theory: recursive calls represent transitions in a decision tree.
Diagram (Mermaid):
flowchart TD
A["state"] --> B["choose option"]
B --> C["recurse"]
C --> D["combine result"]
Worked Example: factorial and simple depth-first traversal.
Common Mistakes: unclear base cases leading to infinite recursion.
Recap:
- state and base case first
- recursion tree explains complexity
- return contract must be explicit
Practice:
- define recursion template for sum of tree nodes`
            },
            {
                title: "Lesson 1: Backtracking with Undo Discipline",
                content: `Why this matters: combinatorial tasks depend on clean choose-recurse-undo logic.
Learning Objective: implement backtracking with predictable state restoration.
Core Theory: each branch mutates state, explores, then rolls back to previous state.
Diagram (Mermaid):
flowchart LR
A["choose"] --> B["recurse"]
B --> C["undo"]
C --> D["next choice"]
Worked Example: subsets generation and combination sum.
Common Mistakes: forgetting to undo mutable state changes.
Recap:
- mutation and rollback are paired operations
- pruning improves exponential search
- deterministic order helps debugging
Practice:
- generate all subsets of unique numbers`
            },
            {
                title: "Lesson 2: Tree Traversal Orders",
                content: `Why this matters: traversal order determines what information is available when.
Learning Objective: apply preorder, inorder, postorder, and level-order correctly.
Core Theory: preorder for root-first workflows, inorder for BST sorted output, postorder for bottom-up aggregation.
Diagram (Mermaid):
flowchart LR
A["preorder"] --> B["root left right"]
C["inorder"] --> D["left root right"]
E["postorder"] --> F["left right root"]
Worked Example: expression tree evaluation with postorder.
Common Mistakes: assuming inorder sorted output for non-BST trees.
Recap:
- traversal order must match objective
- DFS and BFS have different memory profiles
- level-order uses queue boundaries
Practice:
- produce all traversals for a sample binary tree`
            },
            {
                title: "Lesson 3: BFS on Trees and Graphs",
                content: `Why this matters: BFS gives shortest path in unweighted graphs and level-based tree insights.
Learning Objective: process frontier layers with queue and visited controls.
Core Theory: mark visited when enqueuing to avoid duplicate work.
Diagram (Mermaid):
flowchart TD
A["enqueue start"] --> B["dequeue"]
B --> C["enqueue unvisited neighbors"]
C --> B
Worked Example: shortest path in grid with obstacles.
Common Mistakes: marking visited too late and enqueueing duplicates.
Recap:
- BFS is layer-based exploration
- queue size snapshots define levels
- visited strategy controls complexity
Practice:
- solve minimum depth of binary tree`
            },
            {
                title: "Lesson 4: Graph Representation and Traversal",
                content: `Why this matters: many interviews begin by asking for graph modeling choices.
Learning Objective: build adjacency list from edges and traverse safely.
Core Theory: adjacency list is memory-efficient for sparse graphs; adjacency matrix is simpler but heavier.
Diagram (Mermaid):
flowchart LR
A["edge list"] --> B["adjacency list"]
B --> C["DFS or BFS"]
Worked Example: count connected components in undirected graph.
Common Mistakes: forgetting to add both directions for undirected edges.
Recap:
- choose representation by graph density
- directed vs undirected changes edge insertion
- component counting needs full node sweep
Practice:
- build adjacency list from edge input`,
                code: `from collections import defaultdict

def build_graph(n, edges, directed=False):
    g = defaultdict(list)
    for u, v in edges:
        g[u].append(v)
        if not directed:
            g[v].append(u)
    for i in range(n):
        g[i]
    return g`
            },
            {
                title: "Lesson 5: BST and Heap Intuition",
                content: `Why this matters: interviewers often test tree invariants and priority-based retrieval.
Learning Objective: distinguish BST search ordering from heap priority ordering.
Core Theory: BST supports ordered queries; heap supports repeated min or max extraction.
Diagram (Mermaid):
flowchart LR
A["BST invariant"] --> B["ordered lookup"]
C["heap invariant"] --> D["priority extraction"]
Worked Example: kth smallest in BST versus kth largest with min-heap.
Common Mistakes: expecting sorted traversal output from heap structure.
Recap:
- BST and heap solve different query families
- invariants define valid operations
- complexity depends on tree balance or heap size
Practice:
- compare BST and heap for top-k problems`
            },
            {
                title: "Lesson 6: Complexity and Recursion Depth Trade-offs",
                content: `Why this matters: recursion depth can fail on large skewed inputs.
Learning Objective: choose recursive or iterative traversal based on depth risk.
Core Theory: recursive DFS uses call stack O(height), iterative DFS uses explicit stack with similar asymptotic memory.
Worked Example: skewed tree traversal stack overflow mitigation.
Common Mistakes: ignoring recursion limit for worst-case tree shape.
Recap:
- asymptotic complexity and practical limits both matter
- iterative alternatives can improve safety
- choose method consciously
Practice:
- convert recursive DFS to iterative`
            },
            {
                title: "Lesson 7: Recap and Readiness Check",
                content: `Why this matters: this module unlocks graph, DP, and advanced interview patterns.
Learning Objective: verify you can model and traverse state spaces reliably.
Recap:
- recursion and backtracking as state transitions
- BFS and DFS traversal discipline
- graph and tree invariants drive correctness
Practice:
- run one tree and one graph problem in timed mode`
            }
        ]
    }
];

courseData.dsaInterview = [
    {
        number: "DSA · Module 5",
        title: "Dynamic Programming Without Memorization",
        description: "Derive DP from first principles with state-transition reasoning and robust implementation patterns.",
        duration: "120 min",
        lessons: "8 lessons",
        isNew: true,
        isLocked: false,
        topics: ["DP applicability", "State design", "Memoization", "Tabulation", "Knapsack", "String DP", "Space optimization", "DP debugging"],
        detailedDescription: "This module demystifies DP by focusing on state clarity, transitions, and conversion between recursive and iterative forms.",
        detailedContent: [
            {
                title: "Lesson 0: When DP Applies",
                content: `Why this matters: wrong paradigm choice wastes interview time.
Learning Objective: identify overlapping subproblems and optimal substructure.
Core Theory: DP is suitable when repeated states occur and optimal solution can be built from optimal subsolutions.
Diagram (Mermaid):
flowchart TD
A["Problem"] --> B{overlap present}
B -- yes --> C{optimal substructure}
C -- yes --> D["Use DP"]
Worked Example: climbing stairs and coin change.
Common Mistakes: forcing DP where greedy is sufficient.
Recap:
- DP requires reusable states
- define state before coding
- not every optimization problem is DP
Practice:
- classify five problems as DP or non-DP`
            },
            {
                title: "Lesson 1: State and Transition Design",
                content: `Why this matters: unclear state definitions cause most DP failures.
Learning Objective: express minimal complete state and recurrence relation.
Core Theory: state captures exactly what future decisions need; transition computes current from smaller states.
Diagram (Mermaid):
flowchart LR
A["Define state"] --> B["Write transition"]
B --> C["Set base cases"]
Worked Example: house robber with dp i as best till index i.
Common Mistakes: including unnecessary dimensions that inflate complexity.
Recap:
- minimal state improves clarity and performance
- base cases anchor recurrence
- transitions must respect dependencies
Practice:
- define state for longest increasing subsequence`
            },
            {
                title: "Lesson 2: Memoization to Tabulation",
                content: `Why this matters: interviews may request iterative conversion.
Learning Objective: convert top-down memoized recursion to bottom-up table.
Core Theory: dependency graph from recursion determines loop ordering for tabulation.
Diagram (Mermaid):
flowchart TD
A["memo recursion"] --> B["dependency order"]
B --> C["tabulation loops"]
Worked Example: minimum cost climbing stairs.
Common Mistakes: wrong iteration order breaking dependencies.
Recap:
- memo is easier to derive
- table is often easier to optimize
- dependency direction controls loops
Practice:
- convert Fibonacci memo to iterative`,
                code: `def fib(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b`
            },
            {
                title: "Lesson 3: Knapsack and Subset DP",
                content: `Why this matters: many medium interview questions are knapsack variants.
Learning Objective: apply include-exclude transitions with capacity constraints.
Core Theory: state can represent best value or feasibility for prefix items and capacity.
Diagram (Mermaid):
flowchart LR
A["skip item"] --> C["best state"]
B["take item"] --> C
Worked Example: partition equal subset sum with boolean DP.
Common Mistakes: wrong capacity loop direction in compressed 0-1 DP.
Recap:
- include-exclude is core knapsack idea
- loop direction matters for item reuse
- feasibility DP differs from max-value DP
Practice:
- solve target sum as transformed subset problem`
            },
            {
                title: "Lesson 4: String DP Fundamentals",
                content: `Why this matters: edit and sequence problems are high-frequency interview topics.
Learning Objective: use two-dimensional prefix states for sequence comparisons.
Core Theory: dp i j often means answer for first i chars of one string and first j of another.
Diagram (Mermaid):
flowchart LR
A["prefix i"] --> C["state i j"]
B["prefix j"] --> C
Worked Example: longest common subsequence recurrence.
Common Mistakes: mixing subsequence and substring constraints.
Recap:
- prefix framing is reliable
- base row and column initialization is critical
- understand transition cases clearly
Practice:
- implement edit distance`
            },
            {
                title: "Lesson 5: Space Optimization and Trade-offs",
                content: `Why this matters: memory limits can fail otherwise correct solutions.
Learning Objective: reduce table memory where recurrence uses limited history.
Core Theory: rolling arrays and scalar compression are valid when dependencies are local.
Worked Example: unique paths with one-dimensional DP.
Common Mistakes: overwriting needed previous values.
Recap:
- optimize space after correct recurrence is established
- update order matters in-place
- readability trade-offs should be considered
Practice:
- compress a 2D DP to 1D where possible`
            },
            {
                title: "Lesson 6: DP Debugging Framework",
                content: `Why this matters: DP bugs are often subtle indexing or base-case errors.
Learning Objective: debug DP systematically using small-state tables.
Core Theory: print minimal table slices and verify transition expectations on tiny inputs.
Worked Example: diagnose wrong base condition in coin change.
Common Mistakes: testing only one sample and assuming transition is correct.
Recap:
- debug with tiny controllable inputs
- verify each transition branch
- compare memo and tab outputs for consistency
Practice:
- debug a broken DP implementation from a failing test case`
            },
            {
                title: "Lesson 7: Recap and Interview Drills",
                content: `Why this matters: DP confidence is a major interview differentiator.
Learning Objective: solve DP problems with state-first explanation under time pressure.
Recap:
- identify DP applicability quickly
- define state and transition cleanly
- optimize after correctness
Practice:
- complete one 1D DP and one 2D DP problem timed`
            }
        ]
    },
    {
        number: "DSA · Module 6",
        title: "Advanced Graphs, Greedy, and Union-Find",
        description: "High-value patterns for stronger interview rounds: ordering, shortest paths, and dynamic connectivity.",
        duration: "115 min",
        lessons: "8 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Topological sort", "Shortest path", "Greedy proofs", "Minimum spanning tree", "Union-Find", "Cycle detection", "Weighted trade-offs", "Interview walkthrough"],
        detailedDescription: "This module strengthens algorithm selection and proof quality for graph and greedy-heavy interviews.",
        detailedContent: [
            {
                title: "Lesson 0: Directed Dependencies and Topological Order",
                content: `Why this matters: dependency ordering appears in build systems and workflows.
Learning Objective: detect cycles and produce valid DAG ordering.
Core Theory: Kahn algorithm repeatedly processes nodes with zero indegree.
Diagram (Mermaid):
flowchart TD
A["Compute indegree"] --> B["Queue zero indegree"]
B --> C["Pop and append"]
C --> D["Reduce neighbor indegree"]
D --> B
Worked Example: course schedule ordering.
Common Mistakes: forgetting to verify processed node count for cycle detection.
Recap:
- topo sort requires DAG
- indegree queue method is robust
- processed count validates feasibility
Practice:
- return empty when cycle exists`
            },
            {
                title: "Lesson 1: Shortest Path Strategy Selection",
                content: `Why this matters: shortest path method depends on edge properties.
Learning Objective: choose BFS or Dijkstra based on weighted constraints.
Core Theory: BFS for unweighted edges, Dijkstra for non-negative weighted edges.
Diagram (Mermaid):
flowchart LR
A["Path problem"] --> B{weighted edges}
B -- no --> C["BFS"]
B -- yes --> D["Dijkstra"]
Worked Example: network delay time with priority queue.
Common Mistakes: using Dijkstra with negative edges.
Recap:
- edge model decides algorithm
- relaxation updates best-known distances
- stale heap entries should be skipped
Practice:
- compare BFS and Dijkstra on two scenarios`
            },
            {
                title: "Lesson 2: Greedy Correctness Principles",
                content: `Why this matters: greedy is accepted only with a correctness argument.
Learning Objective: explain exchange argument and cut-property intuition.
Core Theory: local choices are valid only if replacing them in any optimal solution does not worsen result.
Diagram (Mermaid):
flowchart LR
A["Local best choice"] --> B["Exchange argument"]
B --> C["Global optimum preserved"]
Worked Example: interval scheduling by earliest finishing time.
Common Mistakes: assuming greedy works because it feels intuitive.
Recap:
- greedy needs proof sketch
- look for counterexamples early
- justify with structure, not confidence
Practice:
- provide proof idea for activity selection`
            },
            {
                title: "Lesson 3: Union-Find in Practice",
                content: `Why this matters: DSU solves connectivity updates efficiently.
Learning Objective: implement find with path compression and union by rank.
Core Theory: path compression flattens trees, making repeated finds fast.
Diagram (Mermaid):
flowchart TD
A["find roots"] --> B{same root}
B -- no --> C["union by rank"]
Worked Example: redundant connection detection.
Common Mistakes: forgetting rank updates on equal-rank union.
Recap:
- DSU is ideal for merge-only connectivity
- compression plus rank improves efficiency
- track component count when needed
Practice:
- dynamic islands count with DSU`,
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
            },
            {
                title: "Lesson 4: Minimum Spanning Tree Basics",
                content: `Why this matters: MST appears in network design and optimization interviews.
Learning Objective: compare Kruskal and Prim at a high level.
Core Theory: Kruskal sorts edges and unions components; Prim grows from a seed node using frontier edges.
Diagram (Mermaid):
flowchart LR
A["Sort edges"] --> B["Take smallest safe edge"]
B --> C["Union components"]
Worked Example: connect cities with minimal total cost.
Common Mistakes: selecting edges that create cycles.
Recap:
- MST applies to weighted undirected connected graphs
- cycle checks are mandatory
- DSU naturally fits Kruskal
Practice:
- implement Kruskal skeleton`
            },
            {
                title: "Lesson 5: Complexity and Trade-off Table",
                content: `Why this matters: algorithm choice should be explicit and defensible.
Learning Objective: communicate runtime and memory trade-offs among graph methods.
Core Theory: adjacency list traversals are O V plus E; Dijkstra with heap is O V plus E times log V.
Worked Example: choose method for sparse versus dense graph workload.
Common Mistakes: quoting complexity for one representation while using another.
Recap:
- representation impacts complexity
- weighted and directed flags affect method choice
- summarize both time and space
Practice:
- build method-selection table for five graph prompts`
            },
            {
                title: "Lesson 6: End-to-End Interview Simulation",
                content: `Why this matters: integration under pressure is the true challenge.
Learning Objective: run full interview flow for a graph problem.
Core Theory: success requires clear assumptions, algorithm rationale, and focused tests.
Worked Example: solve course schedule with cycle detection and ordering.
Common Mistakes: coding before deciding directed versus undirected model.
Recap:
- model accurately first
- choose algorithm by constraints
- test cycle and disconnected cases
Practice:
- perform one narrated mock on a graph problem`
            },
            {
                title: "Lesson 7: Recap and Drill Set",
                content: `Why this matters: these patterns appear frequently in stronger company rounds.
Learning Objective: identify graph and greedy templates quickly.
Recap:
- topo for dependencies
- Dijkstra for non-negative weighted shortest path
- DSU for dynamic connectivity
Practice:
- solve one topo, one shortest path, and one DSU problem`
            }
        ]
    },
    {
        number: "DSA · Module 7",
        title: "Interview Strategy: Communication, Dry Run, and Trade-offs",
        description: "Operational excellence for coding rounds: clarity, confidence, and low-bug execution.",
        duration: "75 min",
        lessons: "7 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Clarifying questions", "Think-aloud structure", "Timeboxing", "Dry runs", "Complexity narration", "Bug recovery", "Decision trade-offs"],
        detailedDescription: "This module converts algorithm knowledge into repeatable interview performance habits.",
        detailedContent: [
            {
                title: "Lesson 0: The 40-Minute Round Blueprint",
                content: `Why this matters: unmanaged time causes avoidable failures.
Learning Objective: use a fixed round timeline with checkpoints.
Core Theory: divide round into understand, baseline, optimize, implement, test, and summarize phases.
Diagram (Mermaid):
flowchart LR
A["Understand"] --> B["Baseline"]
B --> C["Optimize"]
C --> D["Implement"]
D --> E["Test and summarize"]
Worked Example: minute-by-minute plan for a medium difficulty question.
Common Mistakes: spending too long polishing brute-force solution.
Recap:
- timebox each phase
- keep interviewer aligned with your plan
- preserve final testing window
Practice:
- run a timed 40-minute simulation`
            },
            {
                title: "Lesson 1: Clarifying Questions that Increase Signal",
                content: `Why this matters: good questions prevent wrong assumptions.
Learning Objective: ask concise clarifications on constraints and expected behavior.
Core Theory: clarify input bounds, duplicates, sorting guarantees, mutability, and output tie rules.
Worked Example: tie-breaking policy in top-k and interval overlap problems.
Common Mistakes: asking generic questions without impact on approach.
Recap:
- ask only decision-relevant questions
- restate assumptions before coding
- confirm edge-case expectations
Practice:
- list three clarifying questions for each of five prompts`
            },
            {
                title: "Lesson 2: Think-Aloud Without Noise",
                content: `Why this matters: communication quality strongly affects interviewer trust.
Learning Objective: narrate milestones instead of every line.
Core Theory: speak at transitions: model selection, optimization switch, complexity summary, and test outcomes.
Diagram (Mermaid):
flowchart TD
A["State plan"] --> B["Explain key transition"]
B --> C["Summarize complexity"]
Worked Example: moving from sort-based baseline to hash-based linear pass.
Common Mistakes: long silence or excessive commentary on trivial syntax.
Recap:
- communicate decisions, not keystrokes
- summarize at major checkpoints
- keep language precise and short
Practice:
- record a two-minute think-aloud for one solved problem`
            },
            {
                title: "Lesson 3: Dry Run and Edge-Case Matrix",
                content: `Why this matters: most wrong answers fail on untested boundaries.
Learning Objective: execute a compact but complete test matrix quickly.
Core Theory: always include empty input, single element, duplicates, extremes, and adversarial arrangement.
Diagram (Mermaid):
flowchart LR
A["Happy path"] --> B["Boundary cases"]
B --> C["Adversarial case"]
C --> D["Retest after fixes"]
Worked Example: binary search boundaries on all-equal arrays.
Common Mistakes: testing only prompt example.
Recap:
- test matrix should be habitual
- adversarial cases reveal invariant bugs
- retest after every patch
Practice:
- create a reusable dry-run checklist`
            },
            {
                title: "Lesson 4: Complexity and Trade-off Storytelling",
                content: `Why this matters: a correct solution still needs clear engineering justification.
Learning Objective: present time-space trade-offs and why your choice is practical.
Core Theory: summarize dominant operations, memory overhead, and alternatives considered.
Worked Example: hash map O n memory versus sorting O n log n lower overhead options.
Common Mistakes: saying optimal with no supporting argument.
Recap:
- always include both time and space
- mention assumptions explicitly
- justify approach in context of constraints
Practice:
- write one-paragraph complexity summary for three solved problems`
            },
            {
                title: "Lesson 5: Debug Recovery Under Pressure",
                content: `Why this matters: interviewers observe recovery behavior, not perfection.
Learning Objective: isolate and fix bugs with minimal rewrite.
Core Theory: shrink failing case, identify violated invariant, patch smallest surface, rerun matrix.
Worked Example: off-by-one fix in sliding-window left pointer update.
Common Mistakes: restarting implementation without diagnosing cause.
Recap:
- reduce failing input first
- patch minimally and verify broadly
- narrate recovery calmly
Practice:
- intentionally inject and fix one boundary bug`
            },
            {
                title: "Lesson 6: Recap and Mock Checklist",
                content: `Why this matters: this module turns knowledge into reliable execution.
Learning Objective: run full round protocol consistently.
Recap:
- clarify early
- communicate at milestones
- dry run and complexity summary are mandatory
Practice:
- run two mock rounds with rubric scoring`
            }
        ]
    },
    {
        number: "DSA · Module 8",
        title: "90-Day DSA Roadmap and Practice Engine",
        description: "A measured, outcome-driven training plan from fundamentals to interview readiness.",
        duration: "70 min",
        lessons: "7 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Weekly cadence", "Revision loops", "Pattern rotation", "Mock rounds", "Metrics dashboard", "Weakness targeting", "Final readiness gate"],
        detailedDescription: "This module operationalizes preparation with milestones, feedback loops, and readiness metrics that reflect real interview performance.",
        detailedContent: [
            {
                title: "Lesson 0: Build a Sustainable Weekly Cadence",
                content: `Why this matters: consistency beats irregular high-intensity bursts.
Learning Objective: design a weekly schedule that balances new learning and revision.
Core Theory: a balanced plan includes new problems, spaced review, and mock simulation.
Diagram (Mermaid):
flowchart LR
A["New pattern practice"] --> B["Revision loop"]
B --> C["Mock application"]
C --> A
Worked Example: four days new, two days revision, one day mock.
Common Mistakes: solving only new problems without recall practice.
Recap:
- consistency over intensity
- revision must be scheduled
- mocks convert knowledge to performance
Practice:
- draft your next four-week schedule`
            },
            {
                title: "Lesson 1: Spaced Repetition for Pattern Retention",
                content: `Why this matters: forgetting patterns is the biggest prep frustration.
Learning Objective: apply spaced intervals to solved problem revisits.
Core Theory: revisit solved problems at 3-day, 7-day, and 21-day intervals.
Worked Example: revision tracker with category tags and next-review date.
Common Mistakes: repeating only favorite problem types.
Recap:
- schedule revisits explicitly
- use category-balanced revision
- retrieval practice is more valuable than reread
Practice:
- tag your solved list and generate revisit dates`
            },
            {
                title: "Lesson 2: Pattern Rotation Strategy",
                content: `Why this matters: interviews mix categories unpredictably.
Learning Objective: rotate arrays, trees, graphs, and DP across the week.
Core Theory: interleaving improves transfer and reduces category-specific complacency.
Diagram (Mermaid):
flowchart TD
A["Array and string"] --> B["Tree and graph"]
B --> C["DP and greedy"]
C --> D["Mixed mock"]
Worked Example: weekly rotation map with difficulty ramp.
Common Mistakes: doing one category for weeks and losing breadth.
Recap:
- rotate categories intentionally
- include mixed sets before interviews
- ramp difficulty gradually
Practice:
- create a two-week interleaved plan`
            },
            {
                title: "Lesson 3: Mock Interview Operating Model",
                content: `Why this matters: realistic simulation reduces interview anxiety and variance.
Learning Objective: run mocks with objective scoring and targeted improvement loops.
Core Theory: evaluate understanding, approach quality, implementation accuracy, communication, and testing completeness.
Diagram (Mermaid):
flowchart LR
A["Mock round"] --> B["Rubric scoring"]
B --> C["Root-cause analysis"]
C --> D["Targeted drills"]
Worked Example: post-mock action list for the weakest two dimensions.
Common Mistakes: counting solved only and ignoring communication quality.
Recap:
- mocks should be measured
- feedback must map to actions
- one weak dimension can block offers
Practice:
- run one mock and fill a five-dimension rubric`
            },
            {
                title: "Lesson 4: Metrics That Actually Predict Readiness",
                content: `Why this matters: vanity metrics can hide real weaknesses.
Learning Objective: track meaningful metrics for progress decisions.
Core Theory: high-signal metrics include first-pass correctness, no-hint solve rate, and average debug recovery time.
Worked Example: weekly dashboard by pattern family.
Common Mistakes: tracking only total solved count.
Recap:
- quality metrics beat volume metrics
- trend by category reveals blind spots
- use data to pick next practice topics
Practice:
- define your weekly readiness dashboard`
            },
            {
                title: "Lesson 5: Final 14-Day Interview Taper Plan",
                content: `Why this matters: final weeks should optimize confidence, not introduce chaos.
Learning Objective: run a final preparation taper with mixed timed sets and targeted revision.
Core Theory: reduce new-topic load and increase timed mixed simulations.
Diagram (Mermaid):
flowchart LR
A["Target weak patterns"] --> B["Timed mixed sets"]
B --> C["Mock and review"]
Worked Example: two-week day-by-day taper structure.
Common Mistakes: starting many new hard topics just before interviews.
Recap:
- taper for stability and confidence
- mixed timed sets should dominate
- preserve sleep and consistency
Practice:
- craft your 14-day final plan`
            },
            {
                title: "Lesson 6: Course Recap and Capstone Checklist",
                content: `Why this matters: a final checklist reduces last-minute uncertainty.
Learning Objective: verify readiness across concepts, coding, and communication.
Core Theory: readiness requires balanced ability in arrays, trees, graphs, DP, and interview execution.
Recap:
- full-spectrum pattern coverage
- reliable dry run and complexity narration
- mock performance trends improving
Practice:
- complete one full mock loop and retrospective`
            }
        ]
    }
];



courseData.systemDesignFoundations = [
    {
        number: "System Design · Module 1",
        title: "System Design Foundations and Requirement Framing",
        description: "Learn how to frame requirements, estimate scale, and choose architecture boundaries before drawing boxes.",
        duration: "110 min",
        lessons: "8 lessons",
        isNew: true,
        isLocked: false,
        topics: ["HLD vs LLD", "Functional and non-functional requirements", "Capacity estimation", "SLA and SLO", "Latency and throughput", "Availability budgets", "Failure domains", "Trade-off language"],
        detailedDescription: "This module builds first-principles design thinking so learners can approach any system design prompt with structure, clarity, and realistic constraints.",
        detailedContent: [
            {
                title: "Lesson 0: What Interviewers Actually Evaluate in System Design",
                content: `Why this matters: most weak answers fail from poor framing, not from missing technology names.
Learning Objective: understand the scoring dimensions of a system design round.
Core Theory: evaluators look for requirement clarity, architecture reasoning, failure handling, scaling decisions, and trade-off communication.
Diagram (Mermaid):
flowchart LR
A[Prompt] --> B[Requirement framing]
B --> C[Architecture draft]
C --> D[Scale and failure analysis]
D --> E[Trade-off summary]
Worked Example: explain why two different architectures can both be valid based on requirements.
Common Mistakes: jumping straight to tools without defining constraints.
Recap:
- framing drives architecture quality
- there is rarely one perfect design
- justify every major decision
Practice:
- outline your 5-step design flow for any new prompt`
            },
            {
                title: "Lesson 1: HLD vs LLD and When to Go Deeper",
                content: `Why this matters: candidates lose time by over-implementing when only high-level decisions are needed.
Learning Objective: distinguish high-level architecture discussion from component-level implementation depth.
Core Theory: HLD defines services, data flow, and scaling boundaries; LLD defines class-level behavior, schemas, and exact contracts.
Diagram (Mermaid):
flowchart TD
A[HLD] --> B[Service boundaries]
A --> C[Data flow]
D[LLD] --> E[Interfaces and schemas]
D --> F[Validation and retries]
Worked Example: chat system HLD plus one LLD deep dive for message delivery worker.
Common Mistakes: spending 20 minutes on schema details before confirming requirements.
Recap:
- start at HLD unless interviewer requests LLD
- use LLD to prove implementation maturity
- keep depth aligned to available time
Practice:
- convert one HLD component into concise LLD notes`
            },
            {
                title: "Lesson 2: Requirement Gathering That Prevents Rework",
                content: `Why this matters: unclear requirements create architecture rework later in the round.
Learning Objective: gather high-impact functional and non-functional requirements quickly.
Core Theory: requirements should include use cases, scale targets, consistency needs, latency goals, durability expectations, and regulatory constraints.
Diagram (Mermaid):
flowchart LR
A[Use cases] --> D[Requirement sheet]
B[Scale targets] --> D
C[Quality constraints] --> D
Worked Example: URL shortener with custom aliases and click analytics creates additional write and read paths.
Common Mistakes: asking vague questions that do not change architecture choices.
Recap:
- ask decision-changing questions first
- quantify targets where possible
- document assumptions explicitly
Practice:
- write ten high-signal clarifying questions for social feed design`
            },
            {
                title: "Lesson 3: Capacity Estimation and Back-of-the-Envelope Math",
                content: `Why this matters: rough numbers determine whether architecture is realistic.
Learning Objective: estimate request rate, storage growth, and network throughput from assumptions.
Core Theory: order-of-magnitude estimates are sufficient for first-pass design decisions.
Diagram (Mermaid):
flowchart TD
A[Traffic assumptions] --> B[RPS and peak multiplier]
B --> C[Storage per day]
C --> D[Infra sizing]
Worked Example: estimate daily storage for 20 million messages per day with metadata overhead.
Common Mistakes: using precise numbers with no assumption transparency.
Recap:
- rough estimates are expected
- include peak traffic multiplier
- sizing informs sharding and caching decisions
Practice:
- estimate bandwidth for 1 million daily active users and 20 requests per day`
            },
            {
                title: "Lesson 4: Latency, Throughput, and Availability Targets",
                content: `Why this matters: architecture quality is defined by service-level goals.
Learning Objective: map SLO targets to technical design choices.
Core Theory: latency and throughput targets shape caching and async boundaries; availability targets shape replication and failover strategy.
Diagram (Mermaid):
flowchart LR
A[SLO targets] --> B[Component constraints]
B --> C[Design decisions]
C --> D[Monitoring and alerts]
Worked Example: p95 read latency target pushes cache and edge delivery decisions.
Common Mistakes: discussing scalability without clear SLO numbers.
Recap:
- SLO drives architecture trade-offs
- p95 and p99 are practical latency views
- monitor what you promise
Practice:
- define SLOs for a notification service`
            },
            {
                title: "Lesson 5: Trade-offs You Must Explain Clearly",
                content: `Why this matters: strong candidates articulate why they chose one compromise over another.
Learning Objective: explain consistency, cost, complexity, and performance trade-offs with precision.
Core Theory: every design balances multiple competing goals; the best design is context-dependent.
Diagram (Mermaid):
flowchart TD
A[Consistency] --> D[Trade-off matrix]
B[Latency] --> D
C[Cost] --> D
D --> E[Context-based choice]
Worked Example: eventual consistency in feed fan-out to reduce write latency.
Common Mistakes: claiming maximum consistency and maximum availability without constraints.
Recap:
- trade-offs are mandatory, not optional
- justify with requirements and scale
- show alternatives briefly
Practice:
- compare two designs for chat history consistency`
            },
            {
                title: "Lesson 6: Failure Domains and Resilience Mindset",
                content: `Why this matters: production systems fail in partial and unexpected ways.
Learning Objective: identify single points of failure and propose practical mitigation.
Core Theory: resilience requires redundancy, retries with backoff, timeouts, circuit breakers, and graceful degradation.
Diagram (Mermaid):
flowchart LR
A[Failure event] --> B[Detection]
B --> C[Isolation]
C --> D[Recovery path]
Worked Example: queue worker outage with dead-letter queue fallback.
Common Mistakes: relying only on retries without idempotency.
Recap:
- design for partial failure
- combine retries with idempotency
- define degraded mode behavior
Practice:
- list failure scenarios for URL redirect service and mitigations`
            },
            {
                title: "Lesson 7: Module Recap and Architecture Checklist",
                content: `Why this matters: a checklist prevents missing critical architecture dimensions.
Learning Objective: apply a repeatable checklist before presenting final design.
Core Theory: requirement coverage, capacity sanity, resilience, and observability should all be verified.
Recap:
- requirements before architecture
- estimates before scaling decisions
- resilience and observability are core design components
Practice:
- perform a checklist review on one previous design answer`
            }
        ]
    },
    {
        number: "System Design · Module 2",
        title: "Core Building Blocks: API, Storage, Cache, Queue, and CDN",
        description: "Understand what each platform building block solves, where it breaks, and how to combine them safely.",
        duration: "125 min",
        lessons: "8 lessons",
        isNew: true,
        isLocked: false,
        topics: ["API design", "SQL vs NoSQL", "Caching patterns", "Queue semantics", "Load balancing", "Service discovery", "CDN usage", "Data lifecycle"],
        detailedDescription: "This module gives decision-level understanding of common backend components and their production trade-offs.",
        detailedContent: [
            {
                title: "Lesson 0: API Contracts and Idempotency",
                content: `Why this matters: API contracts define system behavior boundaries.
Learning Objective: design stable and idempotent APIs for distributed workflows.
Core Theory: idempotency keys prevent duplicate side effects under retries.
Diagram (Mermaid):
flowchart LR
A[Client retry] --> B[Idempotency check]
B --> C[Single committed effect]
Worked Example: payment creation endpoint with idempotency key.
Common Mistakes: retrying non-idempotent writes without protection.
Recap:
- contracts are part of system reliability
- idempotency is essential in distributed systems
- version APIs intentionally
Practice:
- design idempotent create-order endpoint`
            },
            {
                title: "Lesson 1: SQL vs NoSQL by Access Pattern",
                content: `Why this matters: storage choice should follow query and consistency requirements.
Learning Objective: choose relational or non-relational models based on workload.
Core Theory: SQL excels at transactional integrity and joins; NoSQL excels at flexible schema and horizontal partitioning.
Diagram (Mermaid):
flowchart TD
A[Access pattern] --> B{strong relational needs}
B -- yes --> C[SQL]
B -- no --> D[NoSQL]
Worked Example: order transactions in SQL plus product catalog in document store.
Common Mistakes: selecting DB type by trend rather than workload.
Recap:
- start from query pattern
- consistency and transaction needs are key
- polyglot persistence is common
Practice:
- pick DB model for messaging metadata vs event logs`
            },
            {
                title: "Lesson 2: Caching Strategies and Invalidation",
                content: `Why this matters: caches improve latency but can introduce stale data risk.
Learning Objective: apply cache-aside, write-through, and TTL strategies safely.
Core Theory: cache invalidation strategy must align with consistency expectations.
Diagram (Mermaid):
flowchart LR
A[Read request] --> B[Cache lookup]
B -- miss --> C[DB read and cache fill]
B -- hit --> D[Return response]
Worked Example: product detail cache with short TTL and explicit purge on updates.
Common Mistakes: infinite TTL for mutable data.
Recap:
- cache strategy depends on freshness tolerance
- invalidation is a first-class design topic
- monitor hit rate and stale read rate
Practice:
- design cache policy for user profile service`
            },
            {
                title: "Lesson 3: Queues, Pub/Sub, and Delivery Guarantees",
                content: `Why this matters: asynchronous pipelines are essential for decoupled systems.
Learning Objective: distinguish queue semantics from pub-sub fan-out semantics.
Core Theory: queues distribute work; pub-sub broadcasts events to subscribers.
Diagram (Mermaid):
flowchart TD
A[Producer] --> B[Queue or topic]
B --> C[Consumer one]
B --> D[Consumer two]
Worked Example: image upload pipeline with async transcoding and notification events.
Common Mistakes: assuming exactly-once delivery without deduplication design.
Recap:
- choose queue vs pub-sub by communication pattern
- idempotent consumers are mandatory
- dead-letter handling improves reliability
Practice:
- design retry and DLQ policy for failed workers`
            },
            {
                title: "Lesson 4: Load Balancing and Service Discovery",
                content: `Why this matters: horizontal scale requires traffic distribution and dynamic endpoint resolution.
Learning Objective: design L4/L7 balancing and service discovery basics.
Core Theory: stateless services simplify scaling; sticky sessions need careful state strategy.
Diagram (Mermaid):
flowchart LR
A[Clients] --> B[Load balancer]
B --> C[Service instances]
C --> D[Discovery and health checks]
Worked Example: API gateway routing to autoscaled service pool.
Common Mistakes: no health checks before routing decisions.
Recap:
- balancing and discovery are paired concerns
- health checks prevent serving broken instances
- prefer stateless services when possible
Practice:
- define balancing strategy for read-heavy endpoint`
            },
            {
                title: "Lesson 5: CDN and Edge Delivery",
                content: `Why this matters: global traffic needs edge caching for low latency.
Learning Objective: decide what to cache at edge and how to control cache keys.
Core Theory: static and semi-static content benefits most from CDN; dynamic personalized responses need careful cache controls.
Diagram (Mermaid):
flowchart LR
A[Origin] --> B[CDN edge]
B --> C[User region requests]
Worked Example: signed URL access for media content with edge TTL.
Common Mistakes: caching personalized responses without key isolation.
Recap:
- edge caching reduces origin load
- cache key design impacts correctness
- purge strategy matters for hotfixes
Practice:
- design CDN policy for video thumbnails and manifests`
            },
            {
                title: "Lesson 6: Data Lifecycle, Retention, and Archival",
                content: `Why this matters: long-term growth and compliance require lifecycle planning.
Learning Objective: separate hot, warm, and cold data paths.
Core Theory: not all data needs low-latency storage forever; move older data to cheaper tiers.
Worked Example: analytics events moved from hot store to object storage partitions.
Common Mistakes: keeping all historical data in primary transactional database.
Recap:
- tier storage by access frequency
- lifecycle policies reduce cost
- archival strategy should preserve queryability
Practice:
- define retention tiers for messaging application`
            },
            {
                title: "Lesson 7: Module Recap and Decision Matrix",
                content: `Why this matters: architecture speed improves when component choices are structured.
Learning Objective: use a quick matrix to map requirements to building blocks.
Recap:
- API contracts, storage, cache, queue, and CDN must align
- every component has failure and consistency costs
- lifecycle planning avoids future re-architecture
Practice:
- fill a component matrix for social feed backend`
            }
        ]
    },
    {
        number: "System Design · Module 3",
        title: "Scalability and Reliability Patterns",
        description: "Scale systems safely with partitioning, replication, backpressure, observability, and graceful degradation.",
        duration: "130 min",
        lessons: "8 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Horizontal scaling", "Replication", "Sharding", "Consistency models", "Rate limiting", "Backpressure", "Resilience patterns", "Observability"],
        detailedDescription: "This module focuses on how systems behave under load, failure, and growth, and how to design stability controls from the start.",
        detailedContent: [
            {
                title: "Lesson 0: Scaling Dimensions and Bottleneck Discovery",
                content: `Why this matters: scaling without bottleneck analysis leads to expensive but ineffective changes.
Learning Objective: identify CPU, memory, IO, network, and datastore bottlenecks.
Core Theory: bottlenecks move as systems scale; performance work should be measurement-driven.
Diagram (Mermaid):
flowchart LR
A[Load increase] --> B[Bottleneck identification]
B --> C[Targeted scaling]
C --> D[Re-measure]
Worked Example: API latency spike traced to DB connection pool saturation.
Common Mistakes: scaling app servers when database is actual bottleneck.
Recap:
- measure before scaling
- bottlenecks shift over time
- optimize the narrowest point first
Practice:
- propose bottleneck checks for feed generation service`
            },
            {
                title: "Lesson 1: Replication and Read-Write Split",
                content: `Why this matters: replication improves availability and read throughput.
Learning Objective: design primary-replica topology with lag awareness.
Core Theory: asynchronous replicas can lag, creating stale reads if not managed.
Diagram (Mermaid):
flowchart TD
A[Primary write] --> B[Replica sync]
B --> C[Read queries]
Worked Example: route critical consistency reads to primary and tolerant reads to replicas.
Common Mistakes: assuming replica reads are always fresh.
Recap:
- replication improves read scale
- lag impacts consistency guarantees
- route reads by freshness requirement
Practice:
- design read routing policy by endpoint type`
            },
            {
                title: "Lesson 2: Sharding and Partition Strategy",
                content: `Why this matters: single-node storage eventually hits throughput and size limits.
Learning Objective: choose partition keys and plan for rebalance.
Core Theory: good keys distribute load evenly while preserving common query patterns.
Diagram (Mermaid):
flowchart LR
A[Shard key choice] --> B[Data distribution]
B --> C[Hotspot risk]
C --> D[Rebalance strategy]
Worked Example: user-id based sharding with hash suffix to reduce hotspots.
Common Mistakes: choosing timestamp-only key causing recent-data hotspot.
Recap:
- partition key quality is critical
- rebalance should be planned early
- cross-shard queries add complexity
Practice:
- propose shard key options for messaging data`
            },
            {
                title: "Lesson 3: Rate Limiting and Backpressure",
                content: `Why this matters: uncontrolled traffic can cascade failures.
Learning Objective: design user and service-level traffic guards.
Core Theory: token bucket and leaky bucket are common approaches; backpressure should signal upstream callers.
Diagram (Mermaid):
flowchart TD
A[Incoming requests] --> B[Rate limiter]
B -- allowed --> C[Service processing]
B -- limited --> D[Throttle response]
Worked Example: per-tenant API quota plus global circuit breaker.
Common Mistakes: dropping traffic silently without client feedback.
Recap:
- limit at edges and critical internals
- expose retry-after guidance
- monitor rejection rate and saturation
Practice:
- design rate limits for public API and internal workers`
            },
            {
                title: "Lesson 4: Timeouts, Retries, and Circuit Breakers",
                content: `Why this matters: resilience patterns prevent one failure from taking down the system.
Learning Objective: combine timeout, retry with jitter, and circuit breaker strategies.
Core Theory: retries without timeouts or jitter can amplify outages.
Diagram (Mermaid):
flowchart LR
A[Call dependency] --> B{timeout}
B -- fail --> C[Retry with backoff]
C --> D{breaker open}
D -- yes --> E[Fast fail fallback]
Worked Example: payment dependency timeout handling with fallback response.
Common Mistakes: infinite retries and synchronized retry storms.
Recap:
- retries require bounded policy
- jitter reduces synchronized spikes
- circuit breakers protect upstream services
Practice:
- define retry policy for notification delivery`
            },
            {
                title: "Lesson 5: Graceful Degradation and Fallback Paths",
                content: `Why this matters: users prefer partial service over total outage.
Learning Objective: identify non-critical features that can degrade first.
Core Theory: define core and optional capabilities; under stress, preserve core path and disable expensive optional features.
Diagram (Mermaid):
flowchart LR
A[System stress] --> B[Disable non-critical features]
B --> C[Preserve core path]
Worked Example: feed service disabling personalized ranking while retaining chronological fallback.
Common Mistakes: no explicit degradation plan before incidents.
Recap:
- design degraded mode intentionally
- preserve correctness and safety first
- communicate degraded state clearly
Practice:
- create fallback strategy for chat attachment previews`
            },
            {
                title: "Lesson 6: Observability and Incident Response",
                content: `Why this matters: you cannot improve what you cannot observe.
Learning Objective: define logs, metrics, traces, and alert thresholds tied to user impact.
Core Theory: golden signals include latency, traffic, errors, and saturation; alerts should map to SLO breaches.
Diagram (Mermaid):
flowchart TD
A[Telemetry] --> B[Dashboards]
B --> C[Alerts]
C --> D[Incident response]
Worked Example: p95 latency and error-rate alerting with runbook links.
Common Mistakes: noisy alerts with no actionable ownership.
Recap:
- instrument before scaling crises
- tie alerts to user-impacting objectives
- maintain clear runbooks
Practice:
- define observability plan for URL redirect service`
            },
            {
                title: "Lesson 7: Module Recap and Reliability Checklist",
                content: `Why this matters: reliability should be systematic, not accidental.
Learning Objective: apply a resilience checklist for every architecture proposal.
Recap:
- scale with measurement and bottleneck focus
- protect system with limits and fallbacks
- observability is core architecture, not afterthought
Practice:
- run a reliability review on one prior design`
            }
        ]
    },
    {
        number: "System Design · Module 4",
        title: "Design Documents, Reviews, and Production Rollout",
        description: "Turn architecture ideas into execution-ready artifacts with risk controls and deployment strategy.",
        duration: "100 min",
        lessons: "7 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Design doc structure", "ADR decisions", "API contracts", "Schema evolution", "Rollout plans", "Risk register", "Post-launch review"],
        detailedDescription: "This module teaches how to write design artifacts that align teams, reduce ambiguity, and support safe production delivery.",
        detailedContent: [
            {
                title: "Lesson 0: High-Signal Design Document Structure",
                content: `Why this matters: strong documents accelerate implementation and reduce misalignment.
Learning Objective: produce a concise but complete architecture document.
Core Theory: a good doc covers requirements, assumptions, architecture, data model, API contracts, capacity, risks, and rollout plan.
Diagram (Mermaid):
flowchart TD
A[Requirements] --> B[Architecture]
B --> C[Data and API design]
C --> D[Risk and rollout]
Worked Example: one-page summary plus deep-dive appendix style doc.
Common Mistakes: oversized docs with weak decision traceability.
Recap:
- structure for decision clarity
- capture assumptions and constraints
- include operational plan
Practice:
- draft a design doc skeleton for notification service`
            },
            {
                title: "Lesson 1: Architecture Decision Records",
                content: `Why this matters: teams need explicit record of why key decisions were made.
Learning Objective: capture alternatives, trade-offs, and chosen path in ADR format.
Core Theory: ADR should include context, options considered, decision, and consequences.
Worked Example: SQL vs NoSQL decision ADR for feed metadata.
Common Mistakes: recording decision without rejected alternatives.
Recap:
- decision logs improve maintainability
- alternatives clarify trade-offs
- ADRs aid onboarding and audits
Practice:
- write a short ADR for caching approach`
            },
            {
                title: "Lesson 2: API and Schema Evolution Strategy",
                content: `Why this matters: systems evolve and backward compatibility is often required.
Learning Objective: design versioning and migration paths without service disruption.
Core Theory: additive changes are safer than breaking changes; migration plans need dual-read or dual-write windows where needed.
Diagram (Mermaid):
flowchart LR
A[Current contract] --> B[Additive version]
B --> C[Client migration]
C --> D[Deprecate old path]
Worked Example: adding optional field with default behavior in API response.
Common Mistakes: hard-breaking schema changes without rollout window.
Recap:
- plan compatibility first
- migration needs observability
- deprecation should be staged
Practice:
- design non-breaking update for message payload schema`
            },
            {
                title: "Lesson 3: Design Reviews and Risk Discovery",
                content: `Why this matters: early review catches hidden constraints cheaply.
Learning Objective: run review sessions focused on failure modes and operations.
Core Theory: useful review prompts include bottlenecks, blast radius, rollback path, and ownership clarity.
Worked Example: pre-launch review catches missing retry budget and no DLQ policy.
Common Mistakes: review sessions focused only on happy-path architecture diagrams.
Recap:
- reviews should be adversarial and constructive
- include operations and security perspectives
- convert findings into tracked action items
Practice:
- prepare a review checklist for feed ranking service`
            },
            {
                title: "Lesson 4: Rollout, Canary, and Rollback",
                content: `Why this matters: deployment strategy can determine incident severity.
Learning Objective: stage releases with progressive exposure and clear rollback triggers.
Core Theory: canary rollout reduces risk by validating behavior on small traffic slices first.
Diagram (Mermaid):
flowchart LR
A[Deploy canary] --> B[Monitor key metrics]
B --> C{healthy}
C -- yes --> D[Progressive rollout]
C -- no --> E[Rollback]
Worked Example: 1 percent to 10 percent to 50 percent rollout policy.
Common Mistakes: rollout without explicit success criteria.
Recap:
- define promotion and rollback gates
- monitor user-impact metrics during rollout
- keep rollback simple and tested
Practice:
- write rollout plan for API version migration`
            },
            {
                title: "Lesson 5: Operational Readiness and Ownership",
                content: `Why this matters: successful systems need clear runbooks and ownership.
Learning Objective: define on-call, dashboards, runbooks, and escalation paths.
Core Theory: incident response quality depends on documentation, ownership, and practiced procedures.
Worked Example: runbook for queue lag spike and consumer failure.
Common Mistakes: launching services without runbook coverage.
Recap:
- ownership should be explicit per service
- runbooks reduce mean time to recovery
- on-call readiness is part of design completeness
Practice:
- create operational readiness checklist`
            },
            {
                title: "Lesson 6: Module Recap and Handoff Checklist",
                content: `Why this matters: architecture value is realized only when delivery is controlled.
Learning Objective: complete a production handoff checklist for new systems.
Recap:
- docs should capture decisions and trade-offs
- rollout needs measurable safety gates
- operational readiness is a release requirement
Practice:
- perform handoff review for one case-study design`
            }
        ]
    }
];

courseData.systemDesignAdvanced = [
    {
        number: "System Design · Module 5",
        title: "Case Study: URL Shortener at Scale",
        description: "Design a production-grade short-link platform with analytics, abuse controls, and low-latency redirects.",
        duration: "115 min",
        lessons: "8 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Functional scope", "ID generation", "Read path optimization", "Analytics pipeline", "Abuse prevention", "Data retention", "Multi-region strategy", "Failure handling"],
        detailedDescription: "This module expands a classic design prompt into realistic production concerns including read-heavy optimization and anti-abuse controls.",
        detailedContent: [
            {
                title: "Lesson 0: Requirements and Constraints",
                content: `Why this matters: URL shortener prompts can vary widely in scope.
Learning Objective: scope features that materially change architecture.
Core Theory: custom aliases, expiration, analytics, and abuse controls each add dedicated paths and storage needs.
Diagram (Mermaid):
flowchart TD
A[Create short URL] --> B[Store mapping]
C[Redirect request] --> D[Resolve and redirect]
D --> E[Emit analytics event]
Worked Example: baseline plus analytics-enabled variant comparison.
Common Mistakes: treating redirect and analytics as same latency path.
Recap:
- requirement scope changes architecture shape
- separate latency-critical and async paths
- define abuse constraints early
Practice:
- list requirement tiers for basic vs enterprise URL shortener`
            },
            {
                title: "Lesson 1: ID Generation and Collision Strategy",
                content: `Why this matters: key generation quality affects scale and reliability.
Learning Objective: compare random, counter-based, and hash-based key strategies.
Core Theory: key service should guarantee uniqueness and support high write throughput.
Diagram (Mermaid):
flowchart LR
A[Create request] --> B[Key generation]
B --> C[Collision check]
C --> D[Persist mapping]
Worked Example: base62 encoded monotonic IDs with shard prefixes.
Common Mistakes: key generation tied to one database hotspot.
Recap:
- key design impacts partitioning and storage
- collision handling must be explicit
- consider predictability and abuse vectors
Practice:
- propose key generation for 10k writes per second`
            },
            {
                title: "Lesson 2: Redirect Hot Path Optimization",
                content: `Why this matters: redirect latency is core user experience metric.
Learning Objective: optimize read path with cache and minimal dependencies.
Core Theory: keep redirect path read-only and low-hop; push analytics async.
Diagram (Mermaid):
flowchart LR
A[User request] --> B[Edge cache]
B -- miss --> C[Mapping store]
C --> D[Redirect response]
Worked Example: edge cache plus regional cache fallback.
Common Mistakes: synchronous analytics writes on redirect path.
Recap:
- isolate critical path
- cache aggressively for hot keys
- async side effects improve latency
Practice:
- design redirect path for p95 below 50 milliseconds`
            },
            {
                title: "Lesson 3: Analytics and Event Pipeline",
                content: `Why this matters: analytics should scale independently from redirect traffic.
Learning Objective: design event ingestion and aggregation pipeline.
Core Theory: append-only event streams support flexible downstream aggregation.
Diagram (Mermaid):
flowchart TD
A[Redirect event] --> B[Queue]
B --> C[Stream processor]
C --> D[Analytics store]
Worked Example: daily click counts and geographic aggregates.
Common Mistakes: coupling analytics writes to transaction store.
Recap:
- decouple analytics from serving path
- design for eventual consistency in reporting
- separate raw and aggregated datasets
Practice:
- choose storage model for high-volume click events`
            },
            {
                title: "Lesson 4: Abuse Detection and Safety Controls",
                content: `Why this matters: short-link platforms are frequent abuse targets.
Learning Objective: enforce abuse controls without harming legitimate traffic.
Core Theory: combine rate limits, domain reputation, content scanning hooks, and takedown workflows.
Diagram (Mermaid):
flowchart LR
A[Create request] --> B[Policy checks]
B --> C{allowed}
C -- yes --> D[Issue link]
C -- no --> E[Block and audit]
Worked Example: per-user and per-IP creation throttles with escalation.
Common Mistakes: relying only on post-facto moderation.
Recap:
- prevention and detection both matter
- policy decisions need auditability
- safety controls should be measurable
Practice:
- draft abuse policy tiers for free and paid users`
            },
            {
                title: "Lesson 5: Multi-Region and Disaster Recovery",
                content: `Why this matters: global usage needs region-aware serving and failover.
Learning Objective: plan replication and failover for redirect availability.
Core Theory: read path can be multi-region active-active while write path may remain controlled for consistency.
Worked Example: DNS failover to secondary region during primary outage.
Common Mistakes: no tested failover runbook.
Recap:
- region strategy depends on consistency tolerance
- failover needs drill practice
- define recovery objectives explicitly
Practice:
- propose RTO and RPO targets for redirect service`
            },
            {
                title: "Lesson 6: Cost Model and Lifecycle",
                content: `Why this matters: high read traffic can become expensive without lifecycle control.
Learning Objective: estimate major cost drivers and optimize storage tiers.
Core Theory: hot and cold key behavior should drive caching and archival policies.
Worked Example: archive expired mappings and keep hot cache for active keys.
Common Mistakes: never expiring unused links and analytics detail forever.
Recap:
- cost is a design dimension
- lifecycle policies reduce long-term burden
- optimize by traffic distribution
Practice:
- build monthly cost estimate for 1 billion redirects`
            },
            {
                title: "Lesson 7: Case Study Recap and Interview Script",
                content: `Why this matters: case-study clarity helps in real interview execution.
Learning Objective: summarize full design in a concise narrative.
Recap:
- separate write, read, and analytics paths
- protect platform with abuse controls
- design failover and lifecycle from day one
Practice:
- deliver a 3-minute architecture walkthrough for this case`
            }
        ]
    },
    {
        number: "System Design · Module 6",
        title: "Case Study: Chat and Notification System",
        description: "Design real-time messaging with ordering guarantees, offline delivery, and push fan-out at scale.",
        duration: "125 min",
        lessons: "8 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Realtime gateways", "Connection state", "Message ordering", "Delivery semantics", "Offline sync", "Push notifications", "Presence system", "Backpressure"],
        detailedDescription: "This module covers practical design decisions for chat platforms, balancing latency, ordering guarantees, and mobile delivery constraints.",
        detailedContent: [
            {
                title: "Lesson 0: Scope and Constraints for Messaging Systems",
                content: `Why this matters: messaging design differs for one-to-one, group chat, and enterprise channels.
Learning Objective: identify requirements that impact architecture choices.
Core Theory: ordering guarantees, read receipts, history retention, and multi-device sync define core complexity.
Diagram (Mermaid):
flowchart LR
A[Send message] --> B[Persist]
B --> C[Fan-out]
C --> D[Deliver online]
C --> E[Queue offline]
Worked Example: compare strict ordering per conversation vs eventual ordering across channels.
Common Mistakes: assuming one ordering model fits all chat contexts.
Recap:
- requirements shape consistency and delivery model
- conversation-level guarantees are common
- offline behavior must be explicit
Practice:
- list constraints for enterprise chat vs consumer chat`
            },
            {
                title: "Lesson 1: Real-Time Gateway and Session Management",
                content: `Why this matters: connection layer drives latency and scale efficiency.
Learning Objective: design WebSocket gateway and session routing strategy.
Core Theory: maintain lightweight connection metadata and route messages through broker-backed fan-out.
Diagram (Mermaid):
flowchart TD
A[Client socket] --> B[Gateway]
B --> C[Session store]
B --> D[Message broker]
Worked Example: sticky gateway routing with session lookup fallback.
Common Mistakes: storing heavy user state only in memory without failover strategy.
Recap:
- gateway should remain stateless where practical
- external session metadata improves resilience
- monitor connection churn and heartbeat failures
Practice:
- design heartbeat policy and disconnect timeout`
            },
            {
                title: "Lesson 2: Message Ordering and Deduplication",
                content: `Why this matters: users expect coherent conversation order.
Learning Objective: enforce conversation-level sequencing and idempotent delivery.
Core Theory: assign monotonic sequence numbers per channel and deduplicate by message ID on consumer side.
Diagram (Mermaid):
flowchart LR
A[Incoming message] --> B[Assign sequence]
B --> C[Persist]
C --> D[Deliver with idempotency key]
Worked Example: handling duplicate sends during mobile reconnect.
Common Mistakes: global ordering requirement that hurts scalability.
Recap:
- order scope should be conversation-based
- idempotency is mandatory for retries
- sequence assignment should be deterministic
Practice:
- define sequence strategy for sharded chat rooms`
            },
            {
                title: "Lesson 3: Offline Delivery and History Sync",
                content: `Why this matters: users switch devices and network states constantly.
Learning Objective: design durable storage and cursor-based catch-up.
Core Theory: track per-device or per-user cursor to fetch missed messages efficiently.
Diagram (Mermaid):
flowchart TD
A[User offline] --> B[Persist to store]
B --> C[Update cursor]
C --> D[Sync on reconnect]
Worked Example: unread badge computed from cursor delta.
Common Mistakes: fetching full conversation history on every reconnect.
Recap:
- cursor-based sync is efficient
- unread counts should be derived predictably
- device-specific state needs clear ownership
Practice:
- design reconnection flow for poor mobile networks`
            },
            {
                title: "Lesson 4: Push Notification Pipeline",
                content: `Why this matters: push delivery is external-dependency heavy and failure-prone.
Learning Objective: build reliable push workflow with retries and fallback channels.
Core Theory: notification workers should be idempotent and aware of platform provider limits.
Diagram (Mermaid):
flowchart LR
A[Message event] --> B[Notification queue]
B --> C[Push worker]
C --> D[Provider API]
Worked Example: fallback from push to email for high-priority alerts.
Common Mistakes: retry storms against provider throttling.
Recap:
- isolate push pipeline from core chat path
- handle provider errors with bounded retries
- track delivery status separately
Practice:
- define retry and dead-letter policy for push failures`
            },
            {
                title: "Lesson 5: Presence and Typing Indicators",
                content: `Why this matters: ephemeral signals should be cheap and eventually consistent.
Learning Objective: design lightweight state propagation for presence updates.
Core Theory: ephemeral signals can tolerate occasional loss and should avoid durable writes per event.
Worked Example: Redis-based presence keys with TTL refresh.
Common Mistakes: storing every typing event in long-term database.
Recap:
- ephemeral data needs short-lived storage
- TTL-based presence simplifies cleanup
- eventual consistency is acceptable for indicators
Practice:
- design presence model for 1 million concurrent users`
            },
            {
                title: "Lesson 6: Scale, Backpressure, and Cost Controls",
                content: `Why this matters: fan-out and media attachments can rapidly increase infrastructure cost.
Learning Objective: add backpressure and quota controls for sustainable scale.
Core Theory: bounded queues, per-tenant limits, and payload size controls prevent resource exhaustion.
Worked Example: large group chat fan-out optimization using pull-based fetch for inactive members.
Common Mistakes: naive fan-out to all devices regardless of active state.
Recap:
- apply limits at gateway and worker stages
- optimize fan-out by activity context
- monitor queue lag and drop rates
Practice:
- propose cost controls for high-volume group channels`
            },
            {
                title: "Lesson 7: Case Study Recap and Interview Delivery",
                content: `Why this matters: messaging systems are common interview case studies.
Learning Objective: present design with clear ordering and failure handling narrative.
Recap:
- separate realtime path, persistence path, and notification path
- define ordering and idempotency explicitly
- design offline sync and recovery paths
Practice:
- deliver a 4-minute walkthrough of this architecture`
            }
        ]
    },
    {
        number: "System Design · Module 7",
        title: "Case Study: Social Feed and Ranking System",
        description: "Design scalable feed generation with ranking, caching, fan-out strategy, and consistency management.",
        duration: "125 min",
        lessons: "8 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Feed requirements", "Fan-out models", "Ranking service", "Feature pipelines", "Cache hierarchy", "Cold start", "Abuse and moderation", "Consistency windows"],
        detailedDescription: "This module tackles one of the most complex system design prompts by decomposing feed generation into manageable architecture stages.",
        detailedContent: [
            {
                title: "Lesson 0: Feed Product Modes and Requirements",
                content: `Why this matters: timeline design depends on product goals and engagement model.
Learning Objective: define feed freshness, relevance, and consistency requirements.
Core Theory: chronological and ranked feeds have different compute and consistency trade-offs.
Diagram (Mermaid):
flowchart LR
A[Content events] --> B[Candidate generation]
B --> C[Ranking]
C --> D[Timeline delivery]
Worked Example: mixed strategy with chronological fallback when ranking unavailable.
Common Mistakes: no explicit freshness window target.
Recap:
- feed mode affects architecture deeply
- define freshness and relevance targets
- fallback strategy is required
Practice:
- write requirements for creator-heavy social app feed`
            },
            {
                title: "Lesson 1: Fan-Out on Write vs Fan-Out on Read",
                content: `Why this matters: fan-out strategy determines write and read cost profile.
Learning Objective: choose and justify fan-out model by traffic distribution.
Core Theory: fan-out on write gives low-latency reads but high write amplification; fan-out on read reduces write cost but increases read computation.
Diagram (Mermaid):
flowchart TD
A[New post] --> B{fan-out mode}
B -- write --> C[Push to follower timelines]
B -- read --> D[Compute timeline at request]
Worked Example: hybrid strategy by follower count threshold.
Common Mistakes: one-size-fits-all fan-out policy.
Recap:
- strategy depends on user graph shape
- hybrid policies are practical
- cache policy must match fan-out mode
Practice:
- define threshold rule for celebrity accounts`
            },
            {
                title: "Lesson 2: Candidate Generation and Ranking",
                content: `Why this matters: ranking quality drives user engagement.
Learning Objective: design candidate retrieval and ranking service boundaries.
Core Theory: candidate generation should be fast and broad; ranking can be more expensive but must meet latency budgets.
Diagram (Mermaid):
flowchart LR
A[User request] --> B[Candidate fetch]
B --> C[Feature enrichment]
C --> D[Ranking model]
D --> E[Top N timeline]
Worked Example: two-stage ranking with coarse filter then fine ranker.
Common Mistakes: running heavyweight model on full candidate pool under strict latency targets.
Recap:
- separate candidate and ranking stages
- budget latency per stage
- keep feature freshness measurable
Practice:
- propose ranking pipeline for 1000 candidates per request`
            },
            {
                title: "Lesson 3: Feature Store and Freshness",
                content: `Why this matters: stale features can degrade ranking correctness.
Learning Objective: choose online and offline feature storage paths.
Core Theory: online features support low-latency inference; offline features support training and backfills.
Worked Example: user engagement embeddings refreshed hourly online and daily offline.
Common Mistakes: serving training-only features in realtime path.
Recap:
- feature freshness is a system requirement
- online and offline feature stores have different SLAs
- schema consistency is critical
Practice:
- define freshness targets for three ranking features`
            },
            {
                title: "Lesson 4: Cache Hierarchy and Timeline Delivery",
                content: `Why this matters: feed reads are high volume and latency sensitive.
Learning Objective: design multi-level caching for timelines and metadata.
Core Theory: cache at edge, service, and data layer with clear invalidation strategy.
Diagram (Mermaid):
flowchart LR
A[Edge cache] --> B[Timeline cache]
B --> C[Data fetch]
C --> D[Render response]
Worked Example: cache segmented by user and page cursor.
Common Mistakes: cache key collisions across experiment variants.
Recap:
- cache keys should encode personalization factors
- invalidation policy must be explicit
- monitor cache hit by endpoint and cohort
Practice:
- design cache key for ranked feed page`
            },
            {
                title: "Lesson 5: Cold Start and Diversity Controls",
                content: `Why this matters: new users and sparse profiles need robust defaults.
Learning Objective: design fallback and exploration strategy for low-signal users.
Core Theory: blend popular content, contextual priors, and exploration quotas.
Worked Example: cold-start feed seeded by region and selected interests.
Common Mistakes: static defaults that never adapt.
Recap:
- cold start requires intentional exploration
- diversity controls reduce repetitive timelines
- fallback logic should be measurable
Practice:
- define cold-start strategy for new region launch`
            },
            {
                title: "Lesson 6: Moderation, Safety, and Consistency Windows",
                content: `Why this matters: unsafe content and consistency lag can harm trust.
Learning Objective: integrate moderation checks with ranking and delivery.
Core Theory: moderation can run sync for high-risk content and async for lower risk with corrective actions.
Worked Example: remove flagged content from cache and timeline stores with backfill.
Common Mistakes: no clear SLA for moderation propagation.
Recap:
- safety controls must integrate with feed serving
- consistency windows should be documented
- correction workflows need audit trail
Practice:
- design moderation propagation path and rollback`
            },
            {
                title: "Lesson 7: Case Study Recap and Interview Narrative",
                content: `Why this matters: feed design is a high-complexity interview prompt.
Learning Objective: present feed architecture as clear staged pipeline.
Recap:
- candidate generation and ranking separation
- fan-out strategy by graph shape
- caching, safety, and consistency controls
Practice:
- present a 5-minute feed design summary with trade-offs`
            }
        ]
    },
    {
        number: "System Design · Module 8",
        title: "Case Study: Video Streaming Platform",
        description: "Design upload, transcoding, storage, DRM, CDN delivery, and QoE analytics for global streaming.",
        duration: "135 min",
        lessons: "8 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Upload flow", "Transcoding pipeline", "Manifest design", "DRM and access control", "CDN strategy", "Playback QoE", "Cost optimization", "Incident handling"],
        detailedDescription: "This capstone module combines asynchronous processing, global content delivery, and analytics-driven optimization in one complete architecture.",
        detailedContent: [
            {
                title: "Lesson 0: Product Scope and Streaming Requirements",
                content: `Why this matters: streaming architecture changes significantly by content type and latency goals.
Learning Objective: define VOD vs live requirements and quality targets.
Core Theory: VOD emphasizes processing and delivery efficiency; live adds strict ingestion and latency constraints.
Diagram (Mermaid):
flowchart LR
A[Upload or ingest] --> B[Processing]
B --> C[Storage and manifest]
C --> D[CDN playback]
Worked Example: compare architecture deltas between VOD and near-live streaming.
Common Mistakes: mixing VOD assumptions into live latency targets.
Recap:
- scope first before architecture depth
- latency target changes pipeline choices
- quality metrics should be explicit
Practice:
- define requirements for educational VOD platform`
            },
            {
                title: "Lesson 1: Media Ingestion and Upload Reliability",
                content: `Why this matters: ingest failure directly affects creator trust and user availability.
Learning Objective: design resumable upload and ingest validation paths.
Core Theory: chunked upload and checksum verification improve reliability for unstable networks.
Diagram (Mermaid):
flowchart TD
A[Client upload chunks] --> B[Ingest gateway]
B --> C[Object storage staging]
C --> D[Validation and metadata]
Worked Example: resumable upload with part-level retry.
Common Mistakes: single large upload request with no resume support.
Recap:
- chunking improves reliability
- checksum and metadata validation protect pipeline
- staging area simplifies recovery
Practice:
- design resumable upload API contract`
            },
            {
                title: "Lesson 2: Transcoding and Pipeline Orchestration",
                content: `Why this matters: transcoding is compute-intensive and must scale independently.
Learning Objective: design async transcode workflows with retry and poison-item handling.
Core Theory: workflow orchestration should track job state transitions and retries deterministically.
Diagram (Mermaid):
flowchart LR
A[New media event] --> B[Transcode queue]
B --> C[Worker pool]
C --> D[Output renditions]
D --> E[Publish manifests]
Worked Example: multiple bitrate ladder generation with per-profile job fan-out.
Common Mistakes: synchronous transcoding during upload request.
Recap:
- transcode path must be asynchronous
- state tracking improves operability
- dead-letter handling prevents stuck pipelines
Practice:
- define worker retry policy and max attempts`
            },
            {
                title: "Lesson 3: Manifest and Adaptive Bitrate Delivery",
                content: `Why this matters: adaptive playback quality impacts buffering and user retention.
Learning Objective: design rendition manifests and player selection logic.
Core Theory: clients choose segment bitrate based on throughput and buffer health.
Diagram (Mermaid):
flowchart TD
A[Player requests manifest] --> B[Available renditions]
B --> C[Select bitrate]
C --> D[Fetch next segment]
Worked Example: ABR policy with startup bitrate and ramp-up control.
Common Mistakes: aggressive upswitch causing rebuffer spikes.
Recap:
- ABR balances quality and stability
- manifest correctness is critical
- monitor switch and rebuffer behavior
Practice:
- propose ABR rules for low-bandwidth regions`
            },
            {
                title: "Lesson 4: DRM, Authorization, and Secure Delivery",
                content: `Why this matters: premium content requires secure access and license controls.
Learning Objective: design tokenized playback authorization and DRM key flow.
Core Theory: short-lived signed URLs and license checks reduce unauthorized access risk.
Diagram (Mermaid):
flowchart LR
A[Playback request] --> B[Auth service]
B --> C[Signed token]
C --> D[CDN segment access]
Worked Example: time-bound playback token with device constraints.
Common Mistakes: long-lived unrestricted URLs for premium media.
Recap:
- enforce authorization at playback path
- keep token TTL short
- separate authentication and content authorization concerns
Practice:
- design secure playback flow for subscription tier`
            },
            {
                title: "Lesson 5: CDN and Multi-Region Delivery Strategy",
                content: `Why this matters: global playback quality depends on edge reach and origin architecture.
Learning Objective: choose origin failover and cache warm-up strategy.
Core Theory: multi-CDN and origin shielding can improve resilience and cost control.
Diagram (Mermaid):
flowchart LR
A[Origin storage] --> B[Shield cache]
B --> C[CDN edge]
C --> D[Viewer]
Worked Example: origin failover across two regions with health-based routing.
Common Mistakes: single-region origin with no failover path.
Recap:
- edge strategy drives startup latency
- origin shielding reduces backend load
- failover must be tested
Practice:
- draft multi-region CDN failover plan`
            },
            {
                title: "Lesson 6: QoE Telemetry and Optimization Loop",
                content: `Why this matters: user-perceived quality should guide engineering priorities.
Learning Objective: instrument playback events and derive actionable QoE metrics.
Core Theory: startup time, rebuffer ratio, bitrate stability, and completion rate are core quality signals.
Diagram (Mermaid):
flowchart TD
A[Playback events] --> B[Telemetry pipeline]
B --> C[QoE dashboard]
C --> D[Encoding and delivery tuning]
Worked Example: reducing startup latency by prefetching first segment variants.
Common Mistakes: collecting telemetry without decision loop ownership.
Recap:
- QoE metrics should drive roadmap
- segment-level telemetry improves diagnosis
- optimization loop needs ownership and cadence
Practice:
- define weekly QoE review workflow`
            },
            {
                title: "Lesson 7: Capstone Recap and Production Checklist",
                content: `Why this matters: streaming systems require coordinated decisions across many layers.
Learning Objective: deliver complete architecture summary with failure and cost considerations.
Recap:
- ingest, processing, delivery, and analytics are distinct stages
- reliability and security are core requirements
- QoE feedback loop sustains long-term improvements
Practice:
- present full streaming architecture in a 6-minute interview response`
            }
        ]
    }
];








