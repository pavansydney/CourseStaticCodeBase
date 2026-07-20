// ============================================================
// Programming Languages track content.
// Loaded only on the Courses page (after script.js). It extends the
// existing global `courseData` object with language-specific tracks.
// ============================================================

/* global courseData */

courseData.javaProgramming = [
    {
        number: "Module 1",
        title: "Java Core and Toolchain",
        description: "Build deep Java fundamentals across runtime internals, type-safe coding, classpath setup, and command-line workflow.",
        duration: "80 min",
        lessons: "15 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Introduction and History", "Java Language Evolution", "JDK/JRE/JVM", "Classpath and Packages", "Compilation and Bytecode", "Project Structure", "Variables and Types", "Operators", "Control Flow", "Methods", "Arrays", "Strings", "Enums", "Wrapper Types", "Command Line Build", "Debugging Basics"],
        detailedDescription: "Beginner-first module with stronger depth on runtime model, program structure, and practical coding discipline.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 1
Module Name: Java Foundations and Tooling
Difficulty: Beginner
Estimated Reading Time: 80 min
Estimated Completion Time: 7-8 hours
Prerequisites: None
Learning Objectives:
• explain how source code becomes JVM execution
• write predictable Java logic with strong type awareness
• use CLI tooling to compile, run, and debug small programs
Skills Gained:
• runtime and toolchain literacy
• clean syntax and control-flow implementation
• starter-level debugging and troubleshooting`
            },
                        {
                                title: "Lesson 0: Java Introduction and History",
                                content: `Why this matters: language history explains why Java emphasizes portability, backward compatibility, and enterprise stability.
Learning Objective: understand where Java came from and why it remains widely used.
Core Theory: Java began at Sun Microsystems (mid-1990s) with a "write once, run anywhere" model. Over time it evolved from applets/desktop-heavy usage to backend, cloud, Android, and large-scale enterprise systems.
Diagram (Mermaid):
timeline
    1995 : Java public release
    2004 : Java 5 generics and annotations
    2014 : Java 8 lambdas and streams
    2023 : Java 21 LTS
Common Mistakes: learning syntax without runtime/tooling context; assuming Java is only legacy enterprise tech.
Recap:
• Java prioritized portability from day one
• Major releases modernized syntax and APIs
• Current Java remains strong for backend and platform ecosystems
Practice:
• map one Java release feature to a real project benefit`
                        },
            {
                title: "Lesson 1: Why Java, and How the Runtime Actually Works",
                content: `Why this matters: Java jobs expect runtime understanding, not just syntax memorization. Knowing bytecode and JVM responsibilities helps you debug startup and compatibility issues quickly.
Learning Objective: understand the Java execution pipeline end-to-end.
Core Theory: Java source (.java) is compiled by javac into bytecode (.class). The JVM loads classes through class loaders, verifies bytecode safety, and executes with interpreter + JIT compilation. JDK is a development kit (compiler + tools), while JRE is runtime components.
Diagram (Mermaid):
flowchart LR
    A[input] --> B[validate]
    B --> C[transform]
    C --> D[format result]
  D --> E[Bytecode Verifier]
  E --> F[JVM Execution + JIT]
Worked Example: run a HelloWorld class and inspect bytecode using javap -c.
Common Mistakes: confusing JDK and JRE; incorrect classpath; class name/file mismatch.
Recap:
• JDK builds programs
• JVM runs bytecode
• bytecode is portable, JVM implementations are platform specific
Practice:
• compile and run HelloWorld from terminal
• run javap -c and identify main method instructions`,
                code: `public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, Java runtime!");
    }
}`
            },
            {
                title: "Lesson 2: Variables, Primitive Types, and Numeric Correctness",
                content: `Why this matters: Many production defects are data-type bugs: truncation, overflow, and wrong numeric representation.
Learning Objective: choose types intentionally and avoid hidden conversions.
Core Theory: primitives have fixed range/precision; widening conversion is safe, narrowing may lose data. int division truncates toward zero. Use BigDecimal for money calculations instead of float/double.
Diagram (Mermaid):
flowchart TD
  A[byte] --> B[short] --> C[int] --> D[long]
  C --> E[float] --> F[double]
Worked Example: bill total with tax using BigDecimal.
Common Mistakes: using double for currency; implicit cast assumptions.
Recap:
• type decisions affect correctness
• numeric operations follow strict conversion rules
• currency needs decimal-safe representation
Practice:
• rewrite an invoice calculator from double to BigDecimal`,
                code: `import java.math.BigDecimal;

BigDecimal price = new BigDecimal("199.99");
BigDecimal qty = new BigDecimal("3");
BigDecimal taxRate = new BigDecimal("0.18");
BigDecimal subtotal = price.multiply(qty);
BigDecimal total = subtotal.add(subtotal.multiply(taxRate));
System.out.println("Total: " + total);`
            },
            {
                title: "Lesson 3: Input, Validation, and Control Flow Patterns",
                content: `Why this matters: Real programs deal with noisy user input, not ideal values.
Learning Objective: build validation-first control flow.
Core Theory: guard clauses reduce nesting and improve readability. Use scanner input with explicit parsing and failure handling.
Diagram (Mermaid):
flowchart LR
  A[Read Input] --> B{Valid?}
  B -- No --> C[Show Error]
  B -- Yes --> D[Process]
Worked Example: grade calculator with boundary validation.
Common Mistakes: skipping input checks; Scanner newline issues.
Recap:
• validate before processing
• guard clauses simplify branching
• failure paths should be explicit
Practice:
• accept 5 scores, reject values outside 0..100, then compute average`,
                code: `import java.util.Scanner;

Scanner sc = new Scanner(System.in);
System.out.print("Enter marks (0-100): ");
int marks = Integer.parseInt(sc.nextLine());
if (marks < 0 || marks > 100) {
    System.out.println("Invalid marks");
    return;
}
String grade = marks >= 90 ? "A" : marks >= 75 ? "B" : marks >= 60 ? "C" : "D";
System.out.println("Grade: " + grade);`
            },
            {
                title: "Lesson 4: Methods, Scope, and Reusability",
                content: `Why this matters: maintainable systems are composed of small, testable methods.
Learning Objective: design method signatures around responsibilities.
Core Theory: parameter list defines contract; return types communicate outcomes; local scope prevents accidental coupling.
Diagram (Mermaid):
flowchart TD
  A[input] --> B[validateMarks]
  B --> C[calculateAverage]
  C --> D[formatReport]
Worked Example: split one long method into three focused helpers.
Common Mistakes: god methods; hidden side effects through mutable shared state.
Recap:
• one method = one primary responsibility
• return useful values instead of printing from deep logic
• keep method names verb-focused
Practice:
• refactor a 60-line main method into 4-6 cohesive methods`
            },
            {
                title: "Lesson 5: Arrays, Strings, Enums, and Wrapper Utilities",
                content: `Why this matters: these are the building blocks for most early and interview coding tasks.
Learning Objective: use core APIs safely and efficiently.
Core Theory: arrays are fixed-size contiguous storage; String is immutable; enums encode finite domain values; wrappers enable parsing and null-aware object use.
Diagram (Mermaid):
flowchart LR
  A[raw input] --> B[String parse]
  B --> C[Wrapper conversion]
  C --> D[Enum mapping]
  D --> E[array processing]
Worked Example: parse status values and count by enum type.
Common Mistakes: off-by-one indexing; assuming String mutation; enum conversion without fallback.
Recap:
• arrays are efficient for index-based access
• strings are immutable by design
• enums increase domain safety
Practice:
• build a parser that maps "NEW/PROCESSING/DONE" into an enum and summarizes counts`,
                code: `enum Status { NEW, PROCESSING, DONE }

String[] raw = {"NEW", "DONE", "NEW"};
int done = 0;
for (String value : raw) {
    Status s = Status.valueOf(value);
    if (s == Status.DONE) done++;
}
System.out.println("Done count: " + done);`
            },
            {
                title: "Lesson 6: CLI Tooling, Build Hygiene, and Debugging Basics",
                content: `Why this matters: teams evaluate engineers by how quickly they can reproduce and diagnose issues.
Learning Objective: compile and run projects predictably from command line.
Core Theory: separate source and output folders, use deterministic commands, and inspect stack traces from top frame to root cause.
Diagram (Mermaid):
flowchart LR
  A[src] --> B[javac -d out]
  B --> C[java -cp out Main]
  C --> D[stack trace analysis]
Worked Example: intentionally trigger NumberFormatException and trace failure path.
Common Mistakes: running stale class files; ignoring first useful stack frame.
Recap:
• reproducible build steps reduce confusion
• stack traces are structured diagnostics
• command-line fluency accelerates troubleshooting
Practice:
• break and fix two runtime errors in a sample CLI app`
            },
            {
                title: "Mini Project: Student Result Analyzer",
                content: `Project Goal: build a console app that reads students and marks, validates input, computes analytics, and prints a clean report.

Minimum Features:
• accept N students with 3-5 subject scores each
• reject invalid marks with clear messages
• compute average, highest scorer, and grade distribution

Quality Expectations:
• methods should be small and reusable
• enum-based grade model
• zero duplicated validation logic`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
• add CSV import support from a local file
• include malformed row reporting without stopping entire import

Success Check:
• valid rows are processed correctly
• invalid rows are counted with reasons`
            },
            {
                title: "Module Quiz",
                content: `1) JVM executes: A) source B) bytecode C) markdown D) jar manifest
2) Best numeric type for currency: A) double B) float C) BigDecimal D) long always
3) String in Java is: A) mutable B) immutable C) numeric D) enum
4) Enum is useful for: A) random values B) finite named constants C) map key only D) file access
5) First step in stack trace debugging: A) guess fix B) read top relevant frame C) restart system D) delete cache`
            },
            {
                title: "Interview Preparation",
                content: `Common prompts:
• explain JDK vs JRE vs JVM clearly
• why BigDecimal for money
• difference between == and equals for String
• debug this stack trace live`
            },
            {
                title: "Module Summary",
                content: `You can now write stable Java basics with runtime awareness, strong typing discipline, and reproducible CLI workflows.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module builds object-oriented design depth and covers composition, contracts, and maintainable class architecture.`
            }
        ]
    },
    {
        number: "Module 2",
        title: "Object-Oriented Design and Architecture",
        description: "Master class modeling, composition, interfaces, dependency inversion, and object contracts used in production Java systems.",
        duration: "85 min",
        lessons: "15 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Classes and Constructors", "Encapsulation", "Immutability", "Inheritance", "Composition Over Inheritance", "Polymorphism", "Interfaces", "Abstract Classes", "Packages", "Access Modifiers", "Object Contracts", "SOLID Basics", "Dependency Injection Basics", "Domain Modeling", "Design Trade-offs"],
        detailedDescription: "Practical OOP module focused on correctness, readability, and extension safety.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 2
Module Name: Object-Oriented Design in Java
Difficulty: Beginner to Intermediate
Estimated Reading Time: 85 min
Estimated Completion Time: 8-9 hours
Prerequisites: Module 1
Learning Objectives:
• model real-world domains with cohesive classes
• apply composition, interfaces, and abstraction appropriately
• implement equals/hashCode/toString contracts correctly
Skills Gained:
• maintainable object design
• contract-driven architecture
• interview-ready OOP reasoning`
            },
            {
                title: "Lesson 1: Class Design, Constructors, and Invariants",
                content: `Why this matters: object bugs usually begin at construction time when invalid state is allowed.
Learning Objective: enforce invariants early.
Core Theory: constructors define required state; invariants are rules that must always hold after object creation.
Diagram (Mermaid):
flowchart LR
  A[new Account] --> B{valid input?}
  B -- No --> C[throw exception]
  B -- Yes --> D[store immutable state]
Worked Example: Account with non-negative opening balance.
Common Mistakes: public fields and post-construction patching.
Recap:
• enforce rules at boundaries
• protect state using private fields
• make invalid states unrepresentable`
            },
            {
                title: "Lesson 2: Encapsulation, Immutability, and API Intent",
                content: `Why this matters: mutable shared state is a major source of defects.
Learning Objective: expose behavior, hide state.
Core Theory: encapsulation controls access; immutability simplifies reasoning and concurrency.
Diagram (Mermaid):
flowchart TD
  A[private state] --> B[behavior methods]
  B --> C[validated updates]
Worked Example: immutable value object for Money.
Common Mistakes: leaking internal collections via getters.
Recap:
• return defensive copies when needed
• prefer immutable value types for shared data
• name methods by domain intent`
            },
            {
                title: "Lesson 3: Inheritance, Polymorphism, and Composition Trade-offs",
                content: `Why this matters: misuse of inheritance creates rigid systems.
Learning Objective: decide between inheritance and composition.
Core Theory: inheritance models "is-a" relation; composition models "has-a" and is usually easier to evolve.
Diagram (Mermaid):
flowchart LR
  A[Service] --> B[depends on PaymentGateway interface]
  B --> C[CardGateway]
  B --> D[UpiGateway]
Worked Example: NotificationService using strategy interface rather than subclass branching.
Common Mistakes: deep inheritance for code reuse only.
Recap:
• default to composition
• keep polymorphic contracts minimal and clear
• avoid inheritance when behavior is optional`
            },
            {
                title: "Lesson 4: Interfaces, Abstract Classes, and SOLID Basics",
                content: `Why this matters: interview and production design both require explicit trade-off decisions.
Learning Objective: choose abstraction style correctly.
Core Theory: interfaces define capability contracts; abstract classes share base implementation; SOLID principles guide maintainability.
Diagram (Mermaid):
flowchart TD
  A[Interface: PaymentGateway] --> B[CardPayment]
  A --> C[WalletPayment]
  D[CheckoutService] --> A
Worked Example: Dependency inversion using interface injection.
Common Mistakes: over-abstracting too early; one implementation with unnecessary interface.
Recap:
• abstractions should solve present variation needs
• DIP improves testability
• SRP keeps classes focused`
            },
            {
                title: "Lesson 5: Object Contracts and Collection Compatibility",
                content: `Why this matters: broken equals/hashCode causes silent data corruption in HashMap/HashSet use cases.
Learning Objective: implement value equality safely.
Core Theory: if equals returns true, hashCode must match; toString should support diagnostics not business logic.
Diagram (Mermaid):
flowchart LR
  A[equal objects] --> B[same hash code]
  B --> C[stable map/set behavior]
Worked Example: Member identity by immutable memberId.
Common Mistakes: mutable fields in hashCode calculation.
Recap:
• define equality on stable identity
• override equals and hashCode together
• use toString for debugging clarity`,
                code: `class Member {
    private final String memberId;

    Member(String memberId) {
        this.memberId = memberId;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof Member)) return false;
        Member other = (Member) obj;
        return memberId.equals(other.memberId);
    }

    @Override
    public int hashCode() {
        return memberId.hashCode();
    }
}`
            },
            {
                title: "Lesson 6: Packages, Layering, and Clean Dependency Direction",
                content: `Why this matters: package structure determines how easily a codebase can evolve.
Learning Objective: organize code by responsibility, not randomness.
Core Theory: common layers are api/app, service, domain/model, repository, and infrastructure. Dependencies should point inward to core business logic.
Diagram (Mermaid):
flowchart TD
  A[app/cli] --> B[service]
  B --> C[domain]
  B --> D[repository interface]
  E[file/sql repository impl] --> D
Worked Example: library management package map.
Common Mistakes: circular package dependencies.
Recap:
• package by feature or responsibility consistently
• keep domain independent from IO details
• clear boundaries reduce rewrites`
            },
            {
                title: "Mini Project: Library Domain Redesign",
                content: `Project Goal: redesign a Library app with strong OOP boundaries and testable service logic.

Required Entities:
• Book, Member, Loan, Catalog

Required Rules:
• cannot issue unavailable book
• cannot exceed member issue limit
• cannot return non-issued book

Quality Bar:
• constructor invariant checks
• equals/hashCode correctness for identity types
• no business logic in CLI classes`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
• add reservation queue and cancellation support
• preserve existing issue/return behavior without regressions`
            },
            {
                title: "Module Quiz",
                content: `1) Prefer composition when relation is: A) is-a B) has-a C) static D) final
2) equals/hashCode mismatch mostly breaks: A) loops B) hash-based collections C) arrays D) streams only
3) SRP means: A) one class one responsibility B) single repo policy C) static return policy D) synchronous runtime processing
4) Best constructor practice: A) allow null then fix later B) validate required fields immediately C) public mutable fields D) no parameters ever
5) Package design should minimize: A) comments B) cohesion C) coupling D) constructors`
            },
            {
                title: "Interview Preparation",
                content: `Interview prompts:
• composition vs inheritance with example
• interface vs abstract class
• design small library or payment model in 10 minutes
• explain equals/hashCode pitfalls`
            },
            {
                title: "Module Summary",
                content: `You can now model domains with clean contracts, clear package boundaries, and extensible object-oriented design.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module deepens collection internals, generic variance, and robust exception strategies for data-heavy workflows.`
            }
        ]
    },
    {
        number: "Module 3",
        title: "Data Structures, Generics, and Reliability",
        description: "Choose the right collection for performance, design type-safe APIs, and model failures with explicit reliability contracts.",
        duration: "90 min",
        lessons: "14 lessons",
        isNew: false,
        isLocked: false,
        topics: ["ArrayList vs LinkedList", "HashMap/TreeMap", "HashSet/TreeSet", "Queue/Deque", "Comparable and Comparator", "Big-O Trade-offs", "Generic Classes", "PECS", "Type Erasure", "Exception Hierarchy", "Checked vs Unchecked", "Custom Exceptions", "Validation Patterns", "Error Propagation", "Resilient Pipelines"],
        detailedDescription: "Intermediate module for scalable in-memory data handling and production-safe failure semantics.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 3
Module Name: Collections, Generics, and Error Handling
Difficulty: Intermediate
Estimated Reading Time: 90 min
Estimated Completion Time: 9-10 hours
Prerequisites: Modules 1-2
Learning Objectives:
• choose data structures by operation profile
• build generic APIs with bounded wildcards
• design clear exception and validation strategy
Skills Gained:
• algorithmic trade-off reasoning
• compile-time type safety patterns
• resilient error handling architecture`
            },
            {
                title: "Lesson 1: Choosing Collections with Complexity Trade-offs",
                content: `Why this matters: performance issues often come from wrong collection choice, not CPU speed.
Learning Objective: map operations to suitable structures.
Core Theory: HashMap offers expected O(1) lookup, TreeMap offers O(log n) sorted lookup, ArrayList has O(1) random access and O(n) middle insert.
Complexity / Trade-offs:
• ArrayList: great reads, expensive middle inserts
• LinkedList: fast edge inserts, slow random access
• HashSet: uniqueness with expected O(1) operations
• TreeSet: ordered uniqueness with O(log n)
Diagram (Mermaid):
flowchart LR
  A[Operation Pattern] --> B{Need ordering?}
  B -- Yes --> C[TreeMap/TreeSet]
  B -- No --> D{Need key lookup?}
  D -- Yes --> E[HashMap]
  D -- No --> F[List/Queue]
Practice:
• redesign a student leaderboard from list scan to map-based indexing`
            },
            {
                title: "Lesson 2: Generics, PECS, and Type Erasure",
                content: `Why this matters: generic misuse leads to brittle APIs and unsafe casts.
Learning Objective: apply variance rules correctly.
Core Theory: PECS means Producer Extends, Consumer Super. Generics are erased at runtime; type checks happen mostly at compile time.
Diagram (Mermaid):
flowchart TD
  A[List<? extends Number>] --> B[read as Number]
  C[List<? super Integer>] --> D[write Integer]
Worked Example: sum producer list and append to consumer list.
Common Mistakes: raw types; wildcard overuse in public APIs.
Practice:
• define copyNumbers(List<? extends Number>, List<? super Number>)`,
                code: `static double sumNumbers(java.util.List<? extends Number> values) {
    double sum = 0.0;
    for (Number n : values) sum += n.doubleValue();
    return sum;
}`
            },
            {
                title: "Lesson 3: Exception Hierarchy and Error Strategy",
                content: `Why this matters: random exception handling makes systems hard to debug and support.
Learning Objective: classify failures and propagate context.
Core Theory: checked exceptions model recoverable scenarios; unchecked exceptions model programming or invariant violations. Wrap low-level exceptions with domain context where necessary.
Diagram (Mermaid):
flowchart LR
  A[IO/DB failure] --> B[Repository Exception]
  B --> C[Service-level Domain Exception]
  C --> D[User-friendly message + log details]
Common Mistakes: catch(Exception) everywhere; swallowing stack traces.
Practice:
• convert generic RuntimeException usage into domain-specific exceptions`
            },
            {
                title: "Lesson 4: Validation Pipelines and Partial Failure Handling",
                content: `Why this matters: batch imports and integration tasks must survive bad records.
Learning Objective: process valid and invalid records deterministically.
Core Theory: split pipeline into parse, validate, transform, persist. Track rejected records separately with reason codes.
Diagram (Mermaid):
flowchart LR
  A[Raw rows] --> B[Parse]
  B --> C{Valid?}
  C -- Yes --> D[Persist]
  C -- No --> E[Reject bucket]
  D --> F[Summary]
  E --> F
Practice:
• implement import that does not stop on malformed rows`
            },
            {
                title: "Lesson 5: Map-Centric Design for Fast Domain Queries",
                content: `Why this matters: product features like search and analytics rely on indexing patterns.
Learning Objective: design in-memory indexes with map + set combinations.
Core Theory: maintain primary map by ID and derived indexes by secondary keys. Keep index updates atomic inside service methods.
Diagram (Mermaid):
flowchart TD
  A[create student] --> B[byId map]
  A --> C[byGrade index]
  A --> D[byCourse index]
Practice:
• create two secondary indexes and keep them synchronized on update`
            },
            {
                title: "Mini Project: Student Record Manager v2",
                content: `Project Goal: build a high-integrity student record manager with indexed queries and robust error reporting.

Required Features:
• primary storage using Map<String, Student>
• grade and course secondary indexes
• import pipeline with reject report
• domain exceptions for missing student, invalid score, duplicate roll number

Evaluation Signals:
• correct collection decisions
• predictable error contracts
• stable output under mixed valid/invalid input`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
• add undo support for last mutation command
• preserve index correctness after undo operations`
            },
            {
                title: "Module Quiz",
                content: `1) Expected average lookup complexity of HashMap: A) O(n) B) O(log n) C) O(1) D) O(n log n)
2) PECS stands for: A) Parse, Execute, Compile, Store B) Producer Extends Consumer Super C) Private Encapsulation Class Scope D) none
3) Type erasure happens at: A) runtime only B) compile stage C) JVM GC D) linker phase
4) Checked exceptions usually represent: A) unrecoverable bugs B) recoverable conditions C) syntax errors D) generics issues
5) Best import strategy for bad rows: A) fail fast on first error B) ignore all errors C) track rejects with reasons D) retry forever`
            },
            {
                title: "Interview Preparation",
                content: `Interview prompts:
• ArrayList vs LinkedList with real workload example
• explain PECS with code
• checked vs unchecked exception policy
• design import pipeline with partial failure`
            },
            {
                title: "Module Summary",
                content: `You can now design data-heavy Java workflows using correct collection trade-offs, safe generic APIs, and explicit error contracts.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module introduces modern Java syntax and functional patterns to write concise, expressive, and testable business logic.`
            }
        ]
    },
    {
        number: "Module 4",
        title: "Modern Java: Streams to Pattern Matching",
        description: "Use Streams, Optional, records, and pattern matching to write concise modern Java without sacrificing readability.",
        duration: "85 min",
        lessons: "13 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Functional Interfaces", "Lambdas", "Method References", "Stream Pipelines", "Collectors", "Advanced Collectors", "Optional Patterns", "java.time", "Records", "Sealed Classes", "Pattern Matching", "Switch Expressions", "Text Blocks", "Refactoring Legacy Loops"],
        detailedDescription: "Intermediate module for expressive Java coding with clear data transformation pipelines.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 4
Module Name: Modern Java and Functional Style
Difficulty: Intermediate
Estimated Reading Time: 85 min
Estimated Completion Time: 8-9 hours
Prerequisites: Modules 1-3
Learning Objectives:
• build readable stream and collector pipelines
• model absence and domain data safely using Optional and records
• apply modern Java language features responsibly
Skills Gained:
• concise transformation logic
• immutable data modeling
• modernization refactoring strategy`
            },
            {
                title: "Lesson 1: Streams Deep Dive (map/filter/reduce/collect)",
                content: `Why this matters: stream pipelines are common in enterprise Java and interview problems.
Learning Objective: compose deterministic pipelines.
Core Theory: intermediate operations are lazy; terminal operations trigger execution; avoid side effects to preserve predictability.
Diagram (Mermaid):
flowchart LR
  A[source] --> B[filter]
  B --> C[map]
  C --> D[groupingBy]
  D --> E[result]
Worked Example: department-wise salary aggregates with collectors.
Common Mistakes: mutating external state inside stream operations.
Practice:
• refactor nested loops into groupingBy + mapping`,
                code: `java.util.Map<String, Double> avgByDept = employees.stream()
    .collect(java.util.stream.Collectors.groupingBy(
        Employee::department,
        java.util.stream.Collectors.averagingDouble(Employee::salary)
    ));`
            },
            {
                title: "Lesson 2: Optional Done Right",
                content: `Why this matters: Optional prevents null bugs only when used with discipline.
Learning Objective: use Optional in return types and flow composition.
Core Theory: Optional is for absent values, not for fields in domain entities in most cases.
Diagram (Mermaid):
flowchart TD
  A[lookup] --> B{found?}
  B -- Yes --> C[Optional.of(value)]
  B -- No --> D[Optional.empty]
Worked Example: manager lookup with fallback and custom exception.
Common Mistakes: Optional.get() without checks; Optional parameters.
Practice:
• replace null checks with map/filter/orElseThrow chain`
            },
            {
                title: "Lesson 3: Records, Sealed Types, and Pattern Matching",
                content: `Why this matters: modern Java can reduce boilerplate while increasing type safety.
Learning Objective: model closed domain hierarchies.
Core Theory: records are immutable carriers; sealed classes constrain inheritance; pattern matching reduces instanceof noise.
Diagram (Mermaid):
flowchart TD
  A[sealed PaymentResult] --> B[Success]
  A --> C[Failure]
  A --> D[Pending]
Worked Example: exhaustive switch over payment outcomes.
Common Mistakes: using records for mutable entities.
Practice:
• convert DTO class hierarchy to sealed + record structure`,
                code: `sealed interface PaymentResult permits Success, Failure {}
record Success(String txnId) implements PaymentResult {}
record Failure(String reason) implements PaymentResult {}

static String message(PaymentResult result) {
    return switch (result) {
        case Success s -> "Paid: " + s.txnId();
        case Failure f -> "Failed: " + f.reason();
    };
}`
            },
            {
                title: "Lesson 4: java.time and Time-Zone Correctness",
                content: `Why this matters: date-time bugs are expensive in global systems.
Learning Objective: represent instants and local business times correctly.
Core Theory: use Instant for timeline events, ZonedDateTime for user-facing time zones, LocalDate for date-only business logic.
Diagram (Mermaid):
flowchart LR
  A[Instant UTC] --> B[Zone conversion]
  B --> C[Local display]
Practice:
• implement monthly report boundaries for two different time zones`
            },
            {
                title: "Lesson 5: Refactoring Imperative Legacy Code",
                content: `Why this matters: real projects require incremental modernization, not rewrites.
Learning Objective: transform legacy loops safely with tests first.
Core Theory: establish behavior baseline, refactor in small steps, keep readability over one-liners.
Diagram (Mermaid):
flowchart LR
  A[legacy method] --> B[characterization test]
  B --> C[incremental refactor]
  C --> D[parity check]
Practice:
• modernize one report method without changing output format`
            },
            {
                title: "Mini Project: Employee Analytics Modernization",
                content: `Project Goal: migrate an imperative employee analytics module to modern Java.

Required Features:
• stream-based department analytics
• Optional-based manager lookup
• record DTO output
• switch expression for category classification

Evaluation Signals:
• unchanged functional behavior
• improved readability and maintainability
• no stream side effects`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
• add rolling 3-month trend using java.time and collectors
• include anomaly detection for missing managers or outlier salaries`
            },
            {
                title: "Module Quiz",
                content: `1) Stream intermediate operations are: A) eager B) lazy C) optional D) mutable
2) Optional best used for: A) every field B) return values with absence C) primitive math D) logging levels
3) Record is ideal for: A) mutable aggregate roots B) immutable DTO/value data C) thread pools D) annotations
4) switch expression improves: A) disk usage B) exhaustive branching clarity C) package loading D) JVM startup
5) Best modernization approach: A) full rewrite B) incremental refactor with tests C) no tests D) random edits`
            },
            {
                title: "Interview Preparation",
                content: `Interview prompts:
• map vs flatMap with Optional and Stream
• reduce vs collect trade-offs
• record vs class decision criteria
• refactor this loop into stream and explain complexity`
            },
            {
                title: "Module Summary",
                content: `You can now apply modern Java features with clear reasoning, balancing conciseness, readability, and correctness.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module moves into concurrency, JVM behavior, IO/NIO, and performance tuning for production-scale systems.`
            }
        ]
    },
    {
        number: "Module 5",
        title: "Concurrency, IO, and JVM Internals",
        description: "Build reliable concurrent workflows, handle file processing efficiently, and reason about JVM behavior under load.",
        duration: "95 min",
        lessons: "14 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Thread Lifecycle", "ExecutorService", "Callable and Future", "CompletableFuture", "Locks and Synchronization", "Concurrent Collections", "Java Memory Model", "IO vs NIO", "File Channels", "Serialization Risks", "JVM Memory Areas", "Garbage Collection", "Profiling Basics", "Logging Strategy", "Configuration and Tuning"],
        detailedDescription: "Advanced-intermediate module for operationally safe Java services and batch systems.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 5
Module Name: Concurrency, IO, and JVM Performance
Difficulty: Intermediate to Advanced
Estimated Reading Time: 95 min
Estimated Completion Time: 10-11 hours
Prerequisites: Modules 1-4
Learning Objectives:
• design thread-safe logic with bounded concurrency
• use modern file APIs for robust data processing
• diagnose memory/performance behavior using JVM fundamentals
Skills Gained:
• race-condition prevention
• async composition patterns
• runtime observability and tuning mindset`
            },
            {
                title: "Lesson 1: Thread Safety Fundamentals and Shared-State Hazards",
                content: `Why this matters: race conditions are difficult to reproduce and expensive in production.
Learning Objective: reason about critical sections and visibility.
Core Theory: synchronization provides mutual exclusion and happens-before guarantees; atomic classes avoid lock-heavy counters.
Diagram (Mermaid):
sequenceDiagram
  participant T1 as Thread A
  participant C as Counter
  participant T2 as Thread B
  T1->>C: read value
  T2->>C: read value
  T1->>C: write value+1
  T2->>C: write value+1 (lost update)
Practice:
• implement counter with synchronized and AtomicInteger, compare behavior`
            },
            {
                title: "Lesson 2: Executors, Futures, and Backpressure Awareness",
                content: `Why this matters: unbounded thread creation crashes services.
Learning Objective: choose bounded pools and task queues.
Core Theory: fixed thread pool protects resources; queue policy controls load; Future/CompletableFuture model async result and error paths.
Diagram (Mermaid):
flowchart LR
  A[tasks] --> B[bounded queue]
  B --> C[fixed pool workers]
  C --> D[future results]
Practice:
• configure fixed pool and timeout strategy for slow tasks`,
                code: `java.util.concurrent.ExecutorService pool = java.util.concurrent.Executors.newFixedThreadPool(4);

java.util.concurrent.CompletableFuture<String> first =
    java.util.concurrent.CompletableFuture.supplyAsync(() -> "A", pool);
java.util.concurrent.CompletableFuture<String> second =
    java.util.concurrent.CompletableFuture.supplyAsync(() -> "B", pool);

String combined = first.thenCombine(second, (a, b) -> a + b).join();
System.out.println(combined);
pool.shutdown();`
            },
            {
                title: "Lesson 3: Concurrent Collections and Locking Strategies",
                content: `Why this matters: synchronized blocks alone do not scale for all workloads.
Learning Objective: use ConcurrentHashMap and read/write lock patterns.
Core Theory: concurrent collections reduce contention; locks should be as narrow as possible.
Diagram (Mermaid):
flowchart TD
  A[updates] --> B[ConcurrentHashMap]
  C[reads] --> B
  D[critical write section] --> E[Lock]
Practice:
• migrate shared HashMap to ConcurrentHashMap with computeIfAbsent`
            },
            {
                title: "Lesson 4: File IO, NIO.2, and Data Integrity",
                content: `Why this matters: file systems fail in partial and unpredictable ways.
Learning Objective: build resilient file processing flows.
Core Theory: java.nio.file APIs support path operations, buffered streaming, and safer file handling; use temp files + atomic move for integrity-sensitive writes.
Diagram (Mermaid):
flowchart LR
  A[input file] --> B[validate line]
  B --> C[write temp output]
  C --> D[atomic move]
Practice:
• implement import that writes to temp file and promotes on success`
            },
            {
                title: "Lesson 5: JVM Memory, GC, and Profiling Basics",
                content: `Why this matters: performance incidents often stem from allocation and retention patterns.
Learning Objective: connect code behavior to memory pressure.
Core Theory: stack stores frames; heap stores objects; GC reclaims unreachable objects. Allocation rate, object lifetime, and large retained graphs influence pause patterns.
Diagram (Mermaid):
flowchart LR
  A[new objects] --> B[young generation]
  B --> C[survivor]
  C --> D[old generation]
  D --> E[major GC]
Practice:
• identify one avoidable allocation hotspot and refactor`
            },
            {
                title: "Lesson 6: Logging, Metrics, and Runtime Configuration",
                content: `Why this matters: debugging production issues without observability is guesswork.
Learning Objective: log meaningful events and expose actionable metrics.
Core Theory: include correlation IDs, operation duration, and outcome status; externalize config via environment or properties.
Diagram (Mermaid):
flowchart LR
  A[event] --> B[structured log]
  B --> C[centralized analysis]
  A --> D[metrics counter/timer]
Practice:
• add latency logging + success/failure counters around batch job processing`
            },
            {
                title: "Mini Project: Multithreaded File Processor v2",
                content: `Project Goal: process large file batches concurrently with deterministic summaries and robust diagnostics.

Required Features:
• bounded ExecutorService with retry for transient read failures
• per-file validation and reject report
• thread-safe aggregate counters
• structured logs with file, line, and reason context

Evaluation Signals:
• no race conditions in summary counts
• resilient behavior under malformed input and partial failures
• reproducible performance profile on repeated runs`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
• add cancellation support and graceful shutdown
• emit top-5 slowest files report at end of run`
            },
            {
                title: "Module Quiz",
                content: `1) Preferred thread management for most apps: A) new Thread everywhere B) ExecutorService C) Timer only D) static global loop
2) Race condition means: A) compile failure B) order-dependent incorrect results C) memory leak always D) no logs
3) ConcurrentHashMap helps with: A) immutable values B) safe concurrent key-value access C) SQL joins D) serialization only
4) NIO.2 primarily improves: A) GUI rendering B) modern file/path handling C) generic variance D) annotations
5) Useful production logs include: A) random print statements B) context + level + outcome C) stack traces only D) no timestamps`
            },
            {
                title: "Interview Preparation",
                content: `Interview prompts:
• explain happens-before in practical terms
• thread pool sizing considerations
• how to debug intermittent race bug
• IO vs NIO trade-offs`
            },
            {
                title: "Module Summary",
                content: `You can now design concurrent Java workflows with stronger reliability, better observability, and informed JVM-performance decisions.`
            },
            {
                title: "Next Module Bridge",
                content: `Final module integrates testing, persistence, build tooling, and architectural delivery through a full capstone.`
            }
        ]
    },
    {
        number: "Module 6",
        title: "Production Java Capstone and Testing",
        description: "Ship a complete Java project with layered architecture, automated tests, persistence design, and interview-grade documentation.",
        duration: "110 min",
        lessons: "15 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Maven and Gradle Basics", "Dependency Management", "JUnit 5", "Mockito", "Test Pyramid", "JDBC Basics", "Transactions", "Repository Pattern", "Configuration Profiles", "Packaging and Deployment", "README and ADR", "Architecture Review", "CI Test Workflow", "Capstone Build", "Final Assessment", "Interview Walkthrough"],
        detailedDescription: "Capstone module that transforms learning into portfolio-ready implementation and communication.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 6
Module Name: Testing, Persistence, and Real-World Capstone
Difficulty: Advanced
Estimated Reading Time: 110 min
Estimated Completion Time: 12-14 hours
Prerequisites: Modules 1-5
Learning Objectives:
• build and test maintainable Java applications end-to-end
• integrate persistence with explicit error and transaction handling
• present architecture and trade-offs confidently in interviews
Skills Gained:
• test-first quality habits
• persistence and repository design
• project storytelling and technical communication`
            },
            {
                title: "Lesson 1: Build Tooling with Maven/Gradle",
                content: `Why this matters: professional Java projects are built, tested, and packaged through standardized build tools.
Learning Objective: understand lifecycle phases and dependency scopes.
Core Theory: compile, test, package phases; test dependencies isolated from runtime; reproducible builds rely on locked versions.
Diagram (Mermaid):
flowchart LR
  A[source] --> B[compile]
  B --> C[test]
  C --> D[package]
  D --> E[artifact]
Practice:
• create minimal build file with JUnit dependency and run tests`
            },
            {
                title: "Lesson 2: Unit Testing with JUnit 5 and Mockito",
                content: `Why this matters: without tests, refactoring becomes risky and slow.
Learning Objective: test service logic with mock dependencies.
Core Theory: unit tests isolate one behavior; mocks verify interaction contracts; assertions should be specific and intention-revealing.
Diagram (Mermaid):
sequenceDiagram
  participant T as Test
  participant S as Service
  participant R as MockRepo
  T->>S: createMember(cmd)
  S->>R: save(member)
  R-->>S: success
  S-->>T: result
Practice:
• write tests for success and failure path of one service method`,
                code: `@org.junit.jupiter.api.Test
void createMember_rejectsDuplicateId() {
    MemberRepository repo = org.mockito.Mockito.mock(MemberRepository.class);
    org.mockito.Mockito.when(repo.existsById("M-1")).thenReturn(true);

    MemberService service = new MemberService(repo);

    org.junit.jupiter.api.Assertions.assertThrows(
        DuplicateMemberException.class,
        () -> service.createMember("M-1", "Asha")
    );
}`
            },
            {
                title: "Lesson 3: Persistence Design with JDBC and Repository Pattern",
                content: `Why this matters: many backend interviews expect at least conceptual JDBC and transaction understanding.
Learning Objective: separate domain logic from data access concerns.
Core Theory: repository interfaces define domain operations; JDBC implementations translate operations into SQL; transactions ensure consistency across multi-step updates.
Diagram (Mermaid):
flowchart TD
  A[Service] --> B[Repository Interface]
  B --> C[JDBC Implementation]
  C --> D[Database]
Practice:
• model two repository methods and map result set to domain objects`
            },
            {
                title: "Lesson 4: Test Strategy and Quality Gates",
                content: `Why this matters: quality is a process, not a final checklist.
Learning Objective: create practical test plans across levels.
Core Theory: test pyramid favors many unit tests, fewer integration tests, and selective end-to-end tests. Include edge and failure cases.
Diagram (Mermaid):
flowchart TD
  A[Unit Tests] --> B[Integration Tests]
  B --> C[End-to-End Tests]
Practice:
• define 10-case test matrix covering happy path, edge case, and invalid input`
            },
            {
                title: "Lesson 5: Architecture Decision Records and Documentation",
                content: `Why this matters: strong engineers explain trade-offs, not just code.
Learning Objective: document architecture choices with consequences.
Core Theory: ADR captures context, decision, options considered, and consequences. README should include setup, run, assumptions, and limitations.
Diagram (Mermaid):
flowchart LR
  A[Requirement] --> B[Option Analysis]
  B --> C[Decision]
  C --> D[Consequences]
Practice:
• write one ADR explaining file-based persistence vs database`
            },
            {
                title: "Capstone Project: Job-Ready Java System",
                content: `Project Goal: deliver a complete Java application with clean architecture, automated tests, and persistence.

Project Options:
• Banking Ledger
• Library Operations Hub
• Inventory and Reorder Platform
• Expense Intelligence Console
• Student Lifecycle Manager

Mandatory Deliverables:
• layered codebase (app/service/domain/repository)
• build tool config with repeatable test command
• at least 15 tests (unit + integration)
• persistence layer with failure handling
• README + one ADR + demo walkthrough script`
            },
            {
                title: "Final Assessment",
                content: `Assessment Format:
• Concept Check: JVM, OOP, collections, generics, concurrency, testing, persistence
• Build Check: implement one feature in existing codebase under constraints
• Debug Check: analyze failing tests and fix root cause
• Architecture Check: defend one design decision and trade-off

Pass Criteria:
• correctness and maintainability
• test coverage for changed behavior
• clear explanation of technical decisions`
            },
            {
                title: "Mini Challenge",
                content: `Implement one extension feature (role-based access, export module, or analytics dashboard layer) without breaking existing tests.`
            },
            {
                title: "Interview Preparation",
                content: `Prepare to answer:
• why this architecture and module split?
• where would you add caching or async processing?
• how do tests protect future refactors?
• what would change when moving from file storage to SQL?`
            },
            {
                title: "Module Summary",
                content: `You now have a portfolio-grade Java implementation path with modern engineering practices and interview-ready articulation.`
            },
            {
                title: "Course Completion Path",
                content: `Complete final assessment, polish capstone artifacts, and prepare a 5-minute architecture walkthrough for interviews.`
            }
        ]
    }
];

courseData.pythonProgramming = [
    {
        number: "Module 1",
        title: "Python Core and Runtime Foundations",
        description: "Build strong Python fundamentals with interpreter internals, syntax discipline, functions, collections, and debugging workflow.",
        duration: "85 min",
        lessons: "14 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Introduction and History", "Python Execution Model", "Interpreter and Bytecode", "Variables and Dynamic Typing", "Operators and Expressions", "Control Flow", "Functions", "Scope Rules", "Strings", "Lists", "Tuples", "Sets", "Dictionaries", "CLI and Debugging Basics", "Code Style"],
        detailedDescription: "Beginner-first module that goes beyond syntax and builds practical understanding of how Python code executes and fails.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 1
Module Name: Python Core and Runtime Foundations
Difficulty: Beginner
Estimated Reading Time: 85 min
Estimated Completion Time: 7-8 hours
Prerequisites: None
Learning Objectives:
- explain how Python source executes through the interpreter
- write reliable control flow and function-based programs
- use core data structures with correct mutability decisions
Skills Gained:
- runtime and debugging literacy
- clean function and data-structure usage
- stronger coding discipline for later modules`
            },
                        {
                                title: "Lesson 0: Python Introduction and History",
                                content: `Why this matters: understanding Python's design goals helps you write idiomatic code instead of treating it like another C-style language.
Learning Objective: understand Python's origins and major ecosystem evolution.
Core Theory: Python was created by Guido van Rossum (early 1990s) with emphasis on readability and developer productivity. Python 3 standardized modern semantics and enabled broad growth across web, automation, data science, and AI.
Diagram (Mermaid):
timeline
    1991 : Python initial release
    2000 : Python 2.x mainstream era
    2008 : Python 3.0 release
    2020 : Python 2 end-of-life
Common Mistakes: writing Python with over-verbose patterns; ignoring Python 3-first ecosystem assumptions.
Recap:
- Python was built for readability and speed of development
- Python 3 is the modern standard
- Ecosystem strength spans scripting, backend, data, and AI
Practice:
- explain one Python design principle and show it in a small code example`
                        },
            {
                title: "Lesson 1: How Python Executes Code",
                content: `Why this matters: strong engineers debug with a runtime mental model, not guesswork.
Learning Objective: understand source -> bytecode -> execution flow.
Core Theory: CPython compiles .py to bytecode and executes via the Python virtual machine. __pycache__ stores cached bytecode artifacts.
Diagram (Mermaid):
flowchart LR
  A[.py source] --> B[compile]
  B --> C[bytecode]
  C --> D[PVM execution]
Worked Example: inspect disassembly of a simple function with dis module.
Common Mistakes: assuming Python is interpreted line-by-line without compilation stage.
Recap:
- Python has a compile step to bytecode
- runtime behavior depends on interpreter implementation
- stack traces map failures to source paths
Practice:
- run dis.dis on one function and explain two opcodes`
            },
            {
                title: "Lesson 2: Variables, Types, and Mutability",
                content: `Why this matters: mutability bugs are among the most common Python pitfalls.
Learning Objective: choose data types and mutation patterns intentionally.
Core Theory: Python names bind to objects; assignment does not copy data by default. Mutable: list/dict/set. Immutable: int/str/tuple/frozenset.
Diagram (Mermaid):
flowchart TD
  A[name a] --> B[object]
  C[name b] --> B
Worked Example: demonstrate shared list reference side effect.
Common Mistakes: using mutable default arguments.
Recap:
- names reference objects
- copying and mutation are separate operations
- immutability simplifies reasoning
Practice:
- fix a function that uses a mutable default list`
            },
            {
                title: "Lesson 3: Control Flow, Guard Clauses, and Exceptions",
                content: `Why this matters: branching quality determines readability and bug rate.
Learning Objective: write explicit decision logic with clean failure paths.
Core Theory: guard clauses reduce nested complexity. Use exceptions for truly exceptional states; use condition checks for expected validation.
Diagram (Mermaid):
flowchart LR
  A[input] --> B{valid?}
  B -- no --> C[return error]
  B -- yes --> D[process]
Common Mistakes: deeply nested if blocks and broad except clauses.
Recap:
- validate early
- keep branches shallow
- separate expected invalid input from unexpected failures`
            },
            {
                title: "Lesson 4: Functions, Scope, and Reusability",
                content: `Why this matters: maintainability depends on small, composable functions.
Learning Objective: design function signatures for clarity and testability.
Core Theory: LEGB scope rules govern name lookup. Prefer pure functions when possible and isolate side effects.
Diagram (Mermaid):
flowchart TD
    A[input] --> B[validate]
    B --> C[transform]
    C --> D[format result]
Common Mistakes: giant functions and hidden global dependencies.
Practice:
- refactor one long script into 4 focused functions`,
                code: `def compute_grade(score: int) -> str:
    if not 0 <= score <= 100:
        raise ValueError("score must be between 0 and 100")
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    return "D"`
            },
            {
                title: "Lesson 5: Core Collections and Access Patterns",
                content: `Why this matters: the right container can simplify code and improve runtime behavior.
Learning Objective: map operations to proper collection types.
Core Theory: list for ordered sequence, dict for key lookup, set for uniqueness, tuple for fixed immutable grouping.
Complexity / Trade-offs:
- dict/set average O(1) membership and lookup
- list O(n) membership scan
- tuple immutable and hashable (if contents are hashable)
Practice:
- redesign a lookup-heavy task from list scan to dict index`
            },
            {
                title: "Lesson 6: CLI Workflow, Tracebacks, and Debugging Basics",
                content: `Why this matters: reproducible command-line runs accelerate troubleshooting.
Learning Objective: run scripts and diagnose errors from tracebacks.
Core Theory: read traceback from last line for error type, then upward for source call path.
Diagram (Mermaid):
flowchart LR
  A[python app.py] --> B[traceback]
  B --> C[identify frame]
  C --> D[fix root cause]
Practice:
- intentionally trigger TypeError and ValueError, then explain traceback frames`
            },
            {
                title: "Mini Project: Student Result Analyzer",
                content: `Project Goal: build a CLI tool that validates marks, computes analytics, and prints report cards.
Required Features:
- read N students and subject scores
- reject invalid rows with clear errors
- output average, grade distribution, and topper
Quality Signals:
- reusable functions
- meaningful variable names
- robust input validation`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
- support both interactive input and CSV input mode
- keep validation behavior identical for both modes
Success Check:
- output parity across modes
- invalid rows include exact reason`
            },
            {
                title: "Module Quiz",
                content: `1) Python name binding means: A) names copy values B) names reference objects C) names are types D) names are immutable
2) Mutable type: A) tuple B) str C) list D) int
3) Best structure for fast key lookup: A) list B) dict C) tuple D) range
4) LEGB stands for: A) Load Execute Guard Break B) Local Enclosed Global Builtins C) Local Eval Guard Base D) none
5) Traceback bottom line usually shows: A) warning only B) exception type/message C) import path D) formatter`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
- mutable vs immutable with examples
- common runtime errors and fixes
- function design for readability
- list vs dict trade-offs`
            },
            {
                title: "Module Summary",
                content: `You can now write and debug reliable Python fundamentals with stronger runtime understanding and data-structure choices.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module moves into advanced function patterns, comprehensions, iterators, generators, and robust file/data handling.`
            }
        ]
    },
    {
        number: "Module 2",
        title: "Functions, Comprehensions, and Data Pipelines",
        description: "Develop expressive Python using comprehensions, iterators, generators, and validation-first data pipeline patterns.",
        duration: "90 min",
        lessons: "15 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Advanced Functions", "*args and **kwargs", "Comprehensions", "Iterator Protocol", "Generators", "Generator Expressions", "Sorting with key", "Collections Module", "CSV and JSON Parsing", "Data Validation", "Error Bucketing", "Functional Helpers", "Pipeline Composition", "Memory-Aware Processing"],
        detailedDescription: "Intermediate module for high-signal data manipulation and memory-efficient processing.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 2
Module Name: Functions, Comprehensions, and Data Pipelines
Difficulty: Intermediate
Estimated Reading Time: 90 min
Estimated Completion Time: 8-9 hours
Prerequisites: Module 1
Learning Objectives:
- design reusable functions with explicit contracts
- build readable transformations with comprehensions and generators
- process CSV/JSON flows with deterministic validation
Skills Gained:
- functional decomposition
- memory-efficient data traversal
- robust parsing and transformation patterns`
            },
            {
                title: "Lesson 1: Advanced Function Contracts",
                content: `Why this matters: clear contracts reduce integration bugs.
Learning Objective: use positional-only, keyword-only, defaults, and unpacking safely.
Core Theory: function signatures document intent and prevent misuse.
Common Mistakes: ambiguous parameters and mutable defaults.
Practice:
- redesign a weak API using keyword-only arguments`
            },
            {
                title: "Lesson 2: Comprehensions without Readability Loss",
                content: `Why this matters: concise code should still be maintainable.
Learning Objective: write single-purpose comprehensions and avoid nesting overload.
Core Theory: list/dict/set comprehensions are best for simple map/filter logic.
Diagram (Mermaid):
flowchart LR
  A[input] --> B[filter]
  B --> C[map]
  C --> D[output]
Practice:
- rewrite loop-based transforms into clean comprehensions`,
                code: `scores = [72, 91, 88, 45, 99]
passed = [s for s in scores if s >= 60]
grade_map = {s: ("A" if s >= 90 else "B" if s >= 75 else "C") for s in passed}`
            },
            {
                title: "Lesson 3: Iterators and Generators for Large Inputs",
                content: `Why this matters: loading everything into memory does not scale.
Learning Objective: process streams lazily.
Core Theory: iterators implement __iter__ and __next__; generators suspend state at yield.
Practice:
- create a generator that yields validated records only`,
                code: `def valid_rows(rows):
    for row in rows:
        if 0 <= row["score"] <= 100:
            yield row`
            },
            {
                title: "Lesson 4: Sorting, Grouping, and Aggregation Patterns",
                content: `Why this matters: analytics tasks require stable and explainable transforms.
Learning Objective: use sorted, key functions, and defaultdict/Counter effectively.
Core Theory: sorting with key avoids custom compare complexity; Counter simplifies frequency stats.
Practice:
- build a ranked leaderboard with tie handling`
            },
            {
                title: "Lesson 5: CSV and JSON Pipelines with Reject Reports",
                content: `Why this matters: production ingestion pipelines must handle imperfect data.
Learning Objective: separate parse, validate, transform, and report stages.
Core Theory: keep a reject bucket with reason codes; never silently drop rows.
Diagram (Mermaid):
flowchart LR
  A[file] --> B[parse]
  B --> C{valid?}
  C -- yes --> D[transform]
  C -- no --> E[reject]
  D --> F[summary]
  E --> F
Practice:
- produce summary and reject CSV in one run`
            },
            {
                title: "Mini Project: Student Record Manager v2",
                content: `Project Goal: build a pipeline-driven student record manager with strong validation and export behavior.
Required Features:
- ingest CSV input
- produce class analytics and error report
- export clean JSON summary
Evaluation Signals:
- deterministic output
- clean stage separation
- accurate reject reasons`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
- add chunked processing for very large files
- include progress logs every N records`
            },
            {
                title: "Module Quiz",
                content: `1) Generator execution is: A) eager B) lazy C) compiled-only D) random
2) Counter is best for: A) object inheritance B) frequency counting C) network retries D) package installs
3) Key benefit of stage-based pipelines: A) fewer files B) testable isolation C) no loops D) less memory always
4) Preferred handling for invalid rows: A) silent ignore B) reject with reason C) crash immediately D) retry forever
5) Comprehensions are best for: A) deeply nested business logic B) short map/filter transforms C) DB schema D) threading`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
- iterator vs generator trade-offs
- CSV ingestion design with partial failures
- readability boundaries for comprehensions
- deterministic data processing principles`
            },
            {
                title: "Module Summary",
                content: `You can now build memory-aware Python data pipelines with strong validation and transformation clarity.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module introduces object-oriented modeling, dataclasses, protocols, and architecture-level design patterns in Python.`
            }
        ]
    },
    {
        number: "Module 3",
        title: "Object-Oriented Python and Design Patterns",
        description: "Build maintainable Python systems using OOP, dataclasses, protocols, composition, and architecture-oriented class design.",
        duration: "95 min",
        lessons: "14 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Class Design", "Dataclasses", "Properties", "Inheritance", "Composition", "Abstract Base Classes", "Protocols", "Dunder Methods", "Class and Static Methods", "Dependency Injection", "Repository Pattern", "Service Layer", "Packaging Layout", "Design Smells"],
        detailedDescription: "Strong OOP and design module that emphasizes extensibility, contracts, and low-coupling architecture.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 3
Module Name: Object-Oriented Python and Design Patterns
Difficulty: Intermediate
Estimated Reading Time: 95 min
Estimated Completion Time: 9-10 hours
Prerequisites: Modules 1-2
Learning Objectives:
- model domains with cohesive classes and clear invariants
- apply composition, abstraction, and protocol-based contracts
- organize code into maintainable architecture layers
Skills Gained:
- practical OOP for real systems
- low-coupling class collaboration
- scalable package structure decisions`
            },
            {
                title: "Lesson 1: Dataclasses, Validation, and Invariants",
                content: `Why this matters: data models must remain valid through lifecycle operations.
Learning Objective: use dataclass with post-init validation.
Core Theory: dataclass reduces boilerplate but invariants still need explicit checks.
Practice:
- create immutable value object with validation`,
                code: `from dataclasses import dataclass

@dataclass(frozen=True)
class Money:
    amount: float
    currency: str

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("amount cannot be negative")`
            },
            {
                title: "Lesson 2: Composition over Inheritance",
                content: `Why this matters: inheritance misuse creates rigid class hierarchies.
Learning Objective: identify when composition is safer and easier to evolve.
Core Theory: favor has-a relationships for behavior assembly.
Practice:
- refactor inheritance-heavy design into strategy composition`
            },
            {
                title: "Lesson 3: Abstract Base Classes and Protocols",
                content: `Why this matters: interfaces improve substitutability and testing.
Learning Objective: model capability contracts with ABC and Protocol.
Core Theory: Protocol enables structural typing and duck-typed contracts.
Practice:
- define repository protocol and two interchangeable implementations`
            },
            {
                title: "Lesson 4: Dunder Methods and Object Contracts",
                content: `Why this matters: domain objects should behave predictably in containers and logs.
Learning Objective: implement __repr__, __eq__, and hashing safely.
Core Theory: equality and hashing must align for set/dict behavior.
Practice:
- create entity identity semantics and verify set behavior`
            },
            {
                title: "Lesson 5: Layered Architecture in Python",
                content: `Why this matters: architecture quality controls change cost.
Learning Objective: separate app, service, repository, and domain layers.
Core Theory: dependency direction should keep domain independent of IO/infrastructure.
Diagram (Mermaid):
flowchart TD
  A[cli/api] --> B[service]
  B --> C[domain]
  B --> D[repository protocol]
  E[file/sql impl] --> D
Practice:
- reorganize flat module into layered package structure`
            },
            {
                title: "Mini Project: Library Domain Redesign",
                content: `Project Goal: redesign a library system with clear OOP boundaries and protocol-driven repositories.
Required Features:
- Book, Member, Loan domain objects
- service rules for issue/return/reservation
- file-backed repository implementation
Quality Signals:
- low coupling between layers
- testable service methods
- clear class responsibilities`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
- add fine policy strategies without changing service API
- add in-memory repository for fast test runs`
            },
            {
                title: "Module Quiz",
                content: `1) Better default for extensibility: A) deep inheritance B) composition C) globals D) metaclasses
2) Protocol primarily supports: A) runtime speed B) structural contracts C) GUI widgets D) SQL joins
3) dataclass(frozen=True) implies: A) mutable fields B) immutable instances C) no constructor D) no typing
4) Layered architecture should minimize: A) cohesion B) boundaries C) coupling D) tests
5) __repr__ is most useful for: A) encryption B) debugging representation C) package install D) API auth`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
- composition vs inheritance examples
- protocol vs abstract base class
- designing testable service layers
- spotting OOP code smells`
            },
            {
                title: "Module Summary",
                content: `You can now design maintainable Python systems with strong domain models, clear contracts, and scalable architecture boundaries.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module adds robust error handling, context managers, logging, and production-grade configuration patterns.`
            }
        ]
    },
    {
        number: "Module 4",
        title: "Error Handling, Context, and Production Practices",
        description: "Strengthen production readiness with structured exceptions, context managers, logging, configuration, and packaging discipline.",
        duration: "90 min",
        lessons: "13 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Exception Hierarchy", "Custom Exceptions", "Context Managers", "with Statement", "Decorators", "Logging Levels", "Structured Logging", "Configuration Loading", "Environment Variables", "Secrets Hygiene", "Type Hints", "Static Analysis", "Packaging Basics"],
        detailedDescription: "Production engineering module that focuses on reliability, observability, and maintainable deployment patterns.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 4
Module Name: Error Handling, Context, and Production Practices
Difficulty: Intermediate
Estimated Reading Time: 90 min
Estimated Completion Time: 8-9 hours
Prerequisites: Modules 1-3
Learning Objectives:
- design explicit exception and recovery boundaries
- use context managers and decorators for safe cross-cutting behavior
- implement logging and configuration patterns used in production
Skills Gained:
- resilient failure modeling
- observability-first coding habits
- safer runtime configuration management`
            },
            {
                title: "Lesson 1: Exception Taxonomy and Recovery Strategy",
                content: `Why this matters: broad catch blocks hide failures and increase MTTR.
Learning Objective: classify errors and attach actionable context.
Core Theory: distinguish validation, dependency, and system failures; map each to clear handling strategy.
Practice:
- convert generic except to targeted exception mapping`
            },
            {
                title: "Lesson 2: Context Managers and Resource Safety",
                content: `Why this matters: leaked resources cause subtle production failures.
Learning Objective: use with and custom context managers for deterministic cleanup.
Core Theory: __enter__/__exit__ define setup and teardown guarantees.
Practice:
- build a custom timer context manager`,
                code: `import time
from contextlib import contextmanager

@contextmanager
def timed(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{label}: {elapsed:.4f}s")`
            },
            {
                title: "Lesson 3: Logging for Debuggability and Operations",
                content: `Why this matters: print debugging does not scale in distributed environments.
Learning Objective: log structured, contextual events.
Core Theory: include operation name, ids, duration, and outcome; avoid sensitive data leakage.
Practice:
- instrument one workflow with info/warn/error and correlation id`
            },
            {
                title: "Lesson 4: Configuration and Secrets Discipline",
                content: `Why this matters: hardcoded settings create deployment and security risks.
Learning Objective: centralize config loading and validation.
Core Theory: read env vars + config file, validate required keys at startup, fail fast for critical settings.
Practice:
- implement typed config object and required-key validation`
            },
            {
                title: "Lesson 5: Typing and Static Quality Checks",
                content: `Why this matters: type hints prevent class of integration bugs before runtime.
Learning Objective: annotate interfaces and enforce checks.
Core Theory: annotate public APIs first; use Optional, Union, and TypedDict where appropriate.
Practice:
- add type hints to service module and resolve static checker warnings`
            },
            {
                title: "Mini Project: Order Processing Service Hardening",
                content: `Project Goal: harden an order processing script into production-safe service module.
Required Features:
- domain-specific exceptions
- context-managed IO and timers
- structured logging with operation ids
- startup config validation
Evaluation Signals:
- predictable error messages
- actionable logs
- clear separation of concerns`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
- add retry policy for transient failures with capped attempts
- emit metrics summary at end of run`
            },
            {
                title: "Module Quiz",
                content: `1) with statement guarantees: A) syntax highlighting B) cleanup execution C) no exceptions D) faster loops
2) Best logging approach: A) print everything B) structured context logs C) no logs D) only stack traces
3) Config best practice: A) hardcode values B) centralized validated loading C) per-function globals D) random defaults
4) Custom exceptions help with: A) color theme B) domain clarity C) package naming D) runtime speed
5) Type hints primarily support: A) readability and tooling checks B) threading C) serialization D) plotting`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
- designing exception boundaries
- context manager implementation use-cases
- logging strategy in production systems
- config and secrets management approach`
            },
            {
                title: "Module Summary",
                content: `You can now engineer Python services with stronger reliability, cleaner operational diagnostics, and safer runtime configuration.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module focuses on concurrency, API integration, testing strategy, and performance profiling for scale-oriented systems.`
            }
        ]
    },
    {
        number: "Module 5",
        title: "Concurrency, APIs, Testing, and Performance",
        description: "Choose the right concurrency model, build resilient API clients, and apply disciplined testing and profiling.",
        duration: "95 min",
        lessons: "14 lessons",
        isNew: false,
        isLocked: false,
        topics: ["GIL and Concurrency Models", "threading", "multiprocessing", "asyncio", "Task Scheduling", "HTTP Clients", "Retries and Timeouts", "Rate Limits", "unittest and pytest Concepts", "Mocking", "Integration Tests", "Profiling", "Benchmarking", "Optimization Trade-offs"],
        detailedDescription: "Advanced module for high-throughput, reliable Python services with measurable quality and performance.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 5
Module Name: Concurrency, APIs, Testing, and Performance
Difficulty: Intermediate to Advanced
Estimated Reading Time: 95 min
Estimated Completion Time: 9-11 hours
Prerequisites: Modules 1-4
Learning Objectives:
- select concurrency model based on workload profile
- design fault-tolerant API integrations with clear retry policies
- build confidence with tests and profile-driven optimization
Skills Gained:
- thread/process/async decision making
- resilient integration patterns
- test and performance engineering mindset`
            },
            {
                title: "Lesson 1: Threads, Processes, and asyncio Trade-offs",
                content: `Why this matters: wrong concurrency model can hurt performance and complexity.
Learning Objective: map IO-bound and CPU-bound tasks correctly.
Core Theory: GIL affects CPU-bound threads; multiprocessing bypasses GIL with process isolation; asyncio excels at IO orchestration.
Diagram (Mermaid):
flowchart LR
  A[Workload] --> B{CPU bound?}
  B -- yes --> C[multiprocessing]
  B -- no --> D{high IO concurrency?}
  D -- yes --> E[asyncio]
  D -- no --> F[threading]
Practice:
- benchmark same task in two concurrency models`
            },
            {
                title: "Lesson 2: Resilient API Client Design",
                content: `Why this matters: external APIs are failure-prone and latency-variable.
Learning Objective: implement timeout, retry, and status-aware handling.
Core Theory: retries should be bounded and applied only to retriable failures; use backoff to reduce thundering herd.
Practice:
- implement client with timeout, retry budget, and reasoned error mapping`
            },
            {
                title: "Lesson 3: Testing Pyramid and Mocking Strategy",
                content: `Why this matters: without tests, refactors become risky.
Learning Objective: split tests into unit, integration, and end-to-end purposefully.
Core Theory: unit tests cover core logic fast; integration tests verify external boundaries; mocks isolate unstable dependencies.
Practice:
- add tests for both success and failure path of API service`,
                code: `import unittest
from unittest.mock import Mock

class TestService(unittest.TestCase):
    def test_retries_on_timeout(self):
        client = Mock()
        client.fetch.side_effect = [TimeoutError(), {"ok": True}]
        service = DataService(client, max_retries=1)
        result = service.load()
        self.assertTrue(result["ok"])`
            },
            {
                title: "Lesson 4: Profiling and Optimization Discipline",
                content: `Why this matters: optimization without measurement often wastes effort.
Learning Objective: locate hotspots with profiling before changing algorithms.
Core Theory: use cProfile and timing metrics; optimize bottlenecks, not every function.
Practice:
- profile one slow pipeline and remove top hotspot`
            },
            {
                title: "Lesson 5: Operational Safeguards for Concurrent Jobs",
                content: `Why this matters: production jobs need cancellation, backpressure, and safe shutdown.
Learning Objective: handle task cancellation and partial failures.
Core Theory: use bounded queues, explicit shutdown paths, and idempotent retry operations.
Practice:
- add graceful cancellation to an async batch worker`
            },
            {
                title: "Mini Project: Async Data Sync Service",
                content: `Project Goal: build a concurrent sync service that fetches remote data, validates it, and stores summarized output.
Required Features:
- asyncio or thread pool based fetch workers
- bounded retry and timeout policy
- unit tests for critical logic paths
- profiling report for one optimization decision
Evaluation Signals:
- stable behavior under intermittent failures
- deterministic retries and logs
- measurable performance improvement`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
- add rate-limit aware request scheduler
- include dead-letter report for permanently failed records`
            },
            {
                title: "Module Quiz",
                content: `1) Best for CPU-bound work in Python: A) threading B) multiprocessing C) asyncio D) recursion
2) Retry policy should be: A) infinite B) bounded and selective C) disabled always D) random
3) Unit tests are best for: A) full deployment checks B) isolated logic behavior C) network latency D) package publishing
4) Profiling should happen: A) after random rewrites B) before optimization C) only in production incidents D) never
5) asyncio is strongest for: A) heavy numeric loops B) high-IO concurrency C) SQL schema design D) static typing`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
- GIL implications and model selection
- retry/backoff design patterns
- mocking strategy in tests
- profiling-driven optimization examples`
            },
            {
                title: "Module Summary",
                content: `You can now build scalable Python integrations with concurrency-aware design, resilient API handling, and measurement-based optimization.`
            },
            {
                title: "Next Module Bridge",
                content: `Final module integrates architecture, persistence, testing, and documentation into a portfolio-ready Python capstone.`
            }
        ]
    },
    {
        number: "Module 6",
        title: "Production Python Capstone",
        description: "Deliver a complete Python system with architecture, persistence, testing, observability, and interview-ready documentation.",
        duration: "115 min",
        lessons: "15 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Requirement Scoping", "Architecture Blueprint", "Domain Modeling", "Persistence Strategy", "Repository Layer", "Service Layer", "Error Contracts", "Config and Logging", "Automated Tests", "Performance Baseline", "Documentation", "ADR Basics", "Final Assessment", "Code Review Checklist", "Demo Narrative"],
        detailedDescription: "Capstone module focused on shipping quality, maintainability, and interview-level articulation.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 6
Module Name: Production Python Capstone
Difficulty: Advanced
Estimated Reading Time: 115 min
Estimated Completion Time: 12-14 hours
Prerequisites: Modules 1-5
Learning Objectives:
- architect and implement an end-to-end Python application
- enforce quality through tests, logging, and structured error handling
- communicate design trade-offs clearly in interview settings
Skills Gained:
- full-project delivery discipline
- production engineering practices
- technical storytelling and review readiness`
            },
            {
                title: "Lesson 1: Scope and Architecture Decisions",
                content: `Why this matters: unmanaged scope is the fastest way to fail capstone delivery.
Learning Objective: define MVP boundaries and architecture choices.
Core Theory: identify entities, workflows, constraints, and non-functional requirements before coding.
Practice:
- write one-page architecture plan with trade-offs`
            },
            {
                title: "Lesson 2: Persistence and Data Integrity",
                content: `Why this matters: persistent state requires consistency guarantees.
Learning Objective: choose file/sql strategy and implement safe write flows.
Core Theory: repository abstraction isolates storage changes; validate-before-write prevents corruption.
Practice:
- implement repository interface with file-backed and in-memory adapter`
            },
            {
                title: "Lesson 3: Test Strategy and Failure Injection",
                content: `Why this matters: robust systems are validated against failure, not just happy paths.
Learning Objective: design test matrix with edge and failure scenarios.
Core Theory: include invalid input, dependency timeout, and partial failure tests.
Practice:
- create 12-case test plan and automate at least 8 cases`
            },
            {
                title: "Lesson 4: Observability and Operational Readiness",
                content: `Why this matters: production support depends on logs and diagnostics quality.
Learning Objective: instrument critical flows with contextual logs and summaries.
Core Theory: log start/end/failure events with ids and duration; avoid sensitive data.
Practice:
- add operation-level logging and run summary metrics`
            },
            {
                title: "Lesson 5: Documentation and Architecture Narrative",
                content: `Why this matters: interviewers evaluate reasoning, not only code output.
Learning Objective: produce clear README, runbook, and ADR-style decision notes.
Core Theory: document alternatives considered, decision rationale, and consequences.
Practice:
- prepare 5-minute architecture walkthrough script`
            },
            {
                title: "Capstone Project Options",
                content: `Choose one project for full delivery.
- Inventory and Reorder Platform
- Student Lifecycle Manager
- Personal Finance Analyzer
- Task Workflow Engine
- API Data Synchronizer
Minimum Delivery Pack:
- layered architecture (app/service/domain/repository)
- automated tests for core behaviors
- persistence with validation
- logs, config, and documentation artifacts`
            },
            {
                title: "Final Assessment",
                content: `Assessment Format:
- concept check across modules 1-6
- implementation challenge with constraints
- debugging exercise on broken workflow
- architecture Q&A and trade-off defense
Pass Criteria:
- correctness and maintainability
- test-backed behavior
- clear technical communication`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
- add one advanced feature (analytics export, role permissions, or caching)
- maintain architectural boundaries and test stability`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
- explain architecture and dependency flow
- justify persistence and error strategy
- describe testing scope and gaps
- deliver concise end-to-end demo narrative`
            },
            {
                title: "Module Summary",
                content: `You now have a production-grade Python capstone path that mirrors real delivery expectations and interview workflows.`
            },
            {
                title: "Course Completion Path",
                content: `Complete final assessment, polish documentation, and prepare your capstone walkthrough for portfolio and interviews.`
            }
        ]
    },
    {
        number: "Module 7",
        title: "Career Specialization Tracks (Optional)",
        description: "Pick one focused track to align Python skills with backend, automation, or data/AI career paths.",
        duration: "80 min",
        lessons: "10 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Backend Track", "Automation Track", "Data and AI Foundations Track", "Portfolio Positioning"],
        detailedDescription: "Optional module that tailors post-capstone learning to specific role goals.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 7
Module Name: Career Specialization Tracks (Optional)
Difficulty: Intermediate to Advanced
Estimated Reading Time: 80 min
Estimated Completion Time: 8-10 hours
Prerequisites: Modules 1-6
Learning Objectives:
- choose a specialization aligned to target role
- build one focused artifact for portfolio depth
- define a 60-day post-course roadmap
Skills Gained:
- role-specific implementation depth
- stronger interview positioning
- personalized upskilling plan`
            },
            {
                title: "Backend Track",
                content: `Core Topics:
- FastAPI or Flask service design
- authentication and authorization basics
- ORM patterns and relational modeling
- async request handling
Artifact: production-style REST API with tests and OpenAPI docs`
            },
            {
                title: "Automation Track",
                content: `Core Topics:
- filesystem and office workflow automation
- schedule-based job execution
- scraping and structured extraction ethics
- operational logging and retry patterns
Artifact: automation suite with dry-run and rollback support`
            },
            {
                title: "Data and AI Foundations Track",
                content: `Core Topics:
- NumPy and Pandas transformations
- exploratory visualization with Matplotlib/Seaborn
- feature preparation basics
- notebook to script production handoff
Artifact: reproducible EDA + model baseline package`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
- deliver one specialization artifact with README, architecture notes, and demo script
Success Check:
- artifact demonstrates role-relevant depth
- roadmap includes measurable next milestones`
            },
            {
                title: "Module Summary",
                content: `This optional module helps you convert core Python competency into a targeted career signal.`
            },
            {
                title: "Course Completion Path",
                content: `Finalize one specialization artifact and add it to your portfolio with your capstone for stronger recruiter visibility.`
            }
        ]
    }
];
courseData.csharpProgramming = [
    {
        number: "Module 1",
        title: "C# and .NET Foundations",
        description: "Build strong C# fundamentals with CLR execution model, type system depth, control flow, and debugging discipline.",
        duration: "85 min",
        lessons: "14 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Introduction and History", "CLR and IL", "JIT Compilation", "Value vs Reference Types", "Variables and Operators", "Control Flow", "Methods", "Parameter Modifiers", "Strings", "Arrays", "Collections Basics", "Namespaces", "Project and Build Basics", "Exception Basics", "Debugging with Stack Traces"],
        detailedDescription: "Beginner-first module that explains how C# actually runs, not just how to write syntax.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 1
Module Name: C# and .NET Foundations
Difficulty: Beginner
Estimated Reading Time: 85 min
Estimated Completion Time: 7-8 hours
Prerequisites: None
Learning Objectives:
- explain source-to-execution flow in .NET
- write type-safe C# logic with predictable behavior
- compile, run, and debug console apps confidently
Skills Gained:
- runtime and toolchain literacy
- C# syntax and type-system confidence
- early debugging and troubleshooting habits`
            },
                        {
                                title: "Lesson 0: C# Introduction and History",
                                content: `Why this matters: C# history explains its balance of productivity, strong typing, and enterprise tooling.
Learning Objective: understand C# evolution and where it fits in modern development.
Core Theory: C# emerged in the early 2000s with .NET to provide modern language features and managed runtime safety. It evolved through generics, LINQ, async/await, records, and cloud-native .NET development.
Diagram (Mermaid):
timeline
    2002 : C# and .NET 1.0
    2005 : Generics in C# 2.0
    2007 : LINQ in C# 3.0
    2012 : async/await in C# 5.0
    2020 : C# 9 records
Common Mistakes: viewing C# as Windows-only legacy stack; overlooking modern cross-platform .NET.
Recap:
- C# evolved continuously with strong language ergonomics
- .NET now supports cross-platform backend and cloud workloads
- Modern C# includes strong async and functional-style tooling
Practice:
- connect one C# feature evolution (for example LINQ or async) to a practical use case`
                        },
            {
                title: "Lesson 1: How C# Runs on .NET",
                content: `Why this matters: runtime understanding helps you diagnose build and execution issues faster.
Learning Objective: understand C# -> IL -> CLR/JIT execution flow.
Core Theory: C# compiles to Intermediate Language (IL). CLR loads assemblies and JIT-compiles IL to machine code at runtime.
Diagram (Mermaid):
flowchart LR
  A[C# source] --> B[compiler]
  B --> C[IL assembly]
  C --> D[CLR loader]
  D --> E[JIT]
  E --> F[native execution]
Worked Example: run a console app and inspect generated assembly metadata.
Common Mistakes: mixing runtime and SDK concepts; targeting wrong framework version.
Recap:
- C# compiles to IL, not machine code directly
- CLR provides execution services
- JIT compiles methods on demand
Practice:
- create, build, and run a new console app from CLI`,
                code: `using System;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Hello from C# runtime");
    }
}`
            },
            {
                title: "Lesson 2: Value Types, Reference Types, and Nullability",
                content: `Why this matters: many C# bugs come from misunderstanding copy behavior and null handling.
Learning Objective: distinguish value vs reference semantics and use nullable annotations correctly.
Core Theory: structs and primitives are value types; class instances are reference types. Nullable reference types help express null contracts.
Diagram (Mermaid):
flowchart TD
  A[value copy] --> B[independent data]
  C[reference copy] --> D[same object]
Worked Example: compare struct copy and class reference behavior.
Common Mistakes: accidental shared mutation through references.
Recap:
- value assignment copies data
- reference assignment copies pointer to object
- nullability annotations improve API clarity`
            },
            {
                title: "Lesson 3: Methods, Parameter Passing, and Scope",
                content: `Why this matters: method contract quality directly affects maintainability.
Learning Objective: design clear signatures using value, ref, out, and optional parameters responsibly.
Core Theory: default passing is by value; ref/out modify caller-visible variables and should be used intentionally.
Practice:
- refactor a method with too many responsibilities into smaller methods`,
                code: `static bool TryParseScore(string input, out int score)
{
    if (int.TryParse(input, out score) && score >= 0 && score <= 100)
    {
        return true;
    }

    score = 0;
    return false;
}`
            },
            {
                title: "Lesson 4: Strings, Arrays, and Basic Collections",
                content: `Why this matters: most business workflows use string processing and collection traversal.
Learning Objective: apply string APIs and choose list/array/dictionary intentionally.
Core Theory: string is immutable; arrays have fixed size; List<T> is dynamic; Dictionary<TKey, TValue> is lookup-focused.
Complexity / Trade-offs:
- Dictionary lookup is expected O(1)
- List index access is O(1), membership scan O(n)
Practice:
- redesign a membership-heavy task from list scan to dictionary`
            },
            {
                title: "Lesson 5: Build Workflow and Debugging Basics",
                content: `Why this matters: reproducible build commands and stack trace literacy save hours.
Learning Objective: compile, run, and diagnose failures predictably.
Core Theory: read exception type and message first, then call stack frames to locate root cause.
Diagram (Mermaid):
flowchart LR
  A[dotnet run] --> B[exception]
  B --> C[stack trace]
  C --> D[root cause fix]
Practice:
- trigger and resolve InvalidOperationException in sample code`
            },
            {
                title: "Mini Project: Student Score Console",
                content: `Project Goal: build a console app that validates marks and prints class analytics.
Required Features:
- input parsing with validation
- grade calculation and summary distribution
- clean method decomposition
Quality Signals:
- clear variable naming
- explicit error messages
- no duplicated logic blocks`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
- support batch input from CSV and interactive mode
- preserve same validation rules for both modes`
            },
            {
                title: "Module Quiz",
                content: `1) C# compiles to: A) native always B) IL C) byte arrays only D) XML
2) Class instances are usually: A) value types B) reference types C) enums D) tuples
3) string in C# is: A) mutable B) immutable C) numeric D) collection only
4) ref/out are mainly for: A) namespace alias B) caller-visible modifications C) async only D) logging
5) First traceback step: A) ignore message B) inspect exception type and frame C) clear bin D) rebuild blindly`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
- CLR vs JIT explanation
- value vs reference type examples
- ref vs out differences
- debugging common runtime exceptions`
            },
            {
                title: "Module Summary",
                content: `You can now write and debug core C# applications with stronger runtime understanding and type-system confidence.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module deepens OOP, interfaces, records, and architecture boundaries used in production C# codebases.`
            }
        ]
    },
    {
        number: "Module 2",
        title: "Object-Oriented C# and Design Principles",
        description: "Master class design, encapsulation, inheritance trade-offs, interfaces, records, and dependency boundaries.",
        duration: "95 min",
        lessons: "15 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Classes and Constructors", "Properties", "Access Modifiers", "Records", "Inheritance", "Composition", "Interfaces", "Abstract Classes", "Polymorphism", "Object Equality", "SOLID Basics", "Dependency Injection Basics", "Namespaces and Assemblies", "Layered Design"],
        detailedDescription: "Core OOP module focused on clean contracts, maintainability, and extension safety.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 2
Module Name: Object-Oriented C# and Design Principles
Difficulty: Beginner to Intermediate
Estimated Reading Time: 95 min
Estimated Completion Time: 9-10 hours
Prerequisites: Module 1
Learning Objectives:
- model domains with cohesive classes and invariants
- apply composition and interfaces for flexible design
- build maintainable dependency boundaries
Skills Gained:
- production-oriented OOP design
- contract-driven architecture
- improved testability mindset`
            },
            {
                title: "Lesson 1: Class Design, Properties, and Invariants",
                content: `Why this matters: invalid object state causes downstream bugs.
Learning Objective: enforce invariants through constructors and controlled setters.
Core Theory: expose behavior, not raw mutable fields. Use properties for validation and intent.
Practice:
- implement Account class that blocks invalid balance updates`
            },
            {
                title: "Lesson 2: Interfaces, Abstraction, and Dependency Inversion",
                content: `Why this matters: interfaces enable substitution and easier testing.
Learning Objective: depend on abstractions, not concrete classes.
Core Theory: high-level modules should not depend directly on low-level details.
Diagram (Mermaid):
flowchart LR
  A[OrderService] --> B[IPaymentGateway]
  B --> C[CardGateway]
  B --> D[WalletGateway]
Practice:
- extract interface from concrete notifier implementation`,
                code: `public interface INotifier
{
    Task SendAsync(string message, CancellationToken ct);
}

public sealed class EmailNotifier : INotifier
{
    public Task SendAsync(string message, CancellationToken ct)
    {
        Console.WriteLine($"Email: {message}");
        return Task.CompletedTask;
    }
}`
            },
            {
                title: "Lesson 3: Composition over Inheritance in C#",
                content: `Why this matters: inheritance is powerful but can over-couple behavior.
Learning Objective: choose composition when behavior should be assembled.
Core Theory: use strategy-like interfaces to swap behavior without deep class hierarchies.
Practice:
- refactor inheritance-heavy flow into composed services`
            },
            {
                title: "Lesson 4: Records, Equality, and Domain Modeling",
                content: `Why this matters: value semantics simplify DTO and immutable data handling.
Learning Objective: use record and class types intentionally.
Core Theory: records provide value-based equality by default; classes usually represent identity-based entities.
Practice:
- convert DTO class set to records and compare behavior`
            },
            {
                title: "Lesson 5: Layered Design and Project Structure",
                content: `Why this matters: architecture determines change cost and testability.
Learning Objective: split app into API, service, domain, and infrastructure layers.
Core Theory: dependency direction should point inward to domain logic.
Diagram (Mermaid):
flowchart TD
  A[API] --> B[Application]
  B --> C[Domain]
  B --> D[Infrastructure Abstractions]
  E[Infrastructure Implementations] --> D
Practice:
- reorganize flat solution into layered folders/projects`
            },
            {
                title: "Mini Project: Library Domain Redesign",
                content: `Project Goal: redesign a library workflow using interface-driven architecture.
Required Features:
- Book, Member, Loan models
- service rules for issue/return
- repository interface with simple in-memory implementation
Evaluation Signals:
- clear boundaries and contracts
- no business logic in presentation layer
- maintainable class responsibilities`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
- add reservation queue and overdue fee strategy
- keep existing issue/return API stable`
            },
            {
                title: "Module Quiz",
                content: `1) Dependency inversion encourages: A) concrete coupling B) interface-driven design C) global state D) inheritance only
2) Records are ideal for: A) mutable entities B) value-centric DTOs C) thread pooling D) reflection only
3) Preferred default for flexibility: A) deep inheritance B) composition C) static classes everywhere D) partial classes
4) Good architecture minimizes: A) cohesion B) coupling C) tests D) interfaces
5) Encapsulation primarily protects: A) package size B) invariants and behavior integrity C) build speed D) syntax`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
- class vs record decisions
- interface and DI benefits
- composition vs inheritance trade-offs
- layered architecture explanation`
            },
            {
                title: "Module Summary",
                content: `You can now design object-oriented C# systems with cleaner boundaries, explicit contracts, and better extension safety.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module covers collections, generics, LINQ, and exception strategy for real data-heavy workflows.`
            }
        ]
    },
    {
        number: "Module 3",
        title: "Collections, Generics, LINQ, and Exceptions",
        description: "Choose data structures by workload, use generics and LINQ safely, and design explicit exception boundaries.",
        duration: "100 min",
        lessons: "14 lessons",
        isNew: false,
        isLocked: false,
        topics: ["List and Dictionary", "HashSet and Queue", "Generic Types and Methods", "Constraints", "LINQ Operators", "Deferred Execution", "Projection and Grouping", "Custom Comparers", "Exception Taxonomy", "Custom Exceptions", "Validation Patterns", "Result Modeling", "Partial Failure Handling", "Data Pipeline Composition"],
        detailedDescription: "Intermediate module for data processing correctness, readability, and resilience.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 3
Module Name: Collections, Generics, LINQ, and Exceptions
Difficulty: Intermediate
Estimated Reading Time: 100 min
Estimated Completion Time: 10-11 hours
Prerequisites: Modules 1-2
Learning Objectives:
- select collections based on operation profile
- build reusable generic APIs and LINQ queries
- model exceptions with consistent recovery behavior
Skills Gained:
- collection trade-off reasoning
- LINQ fluency and pitfalls awareness
- reliability-first error handling patterns`
            },
            {
                title: "Lesson 1: Collection Trade-offs and Complexity",
                content: `Why this matters: wrong collection choices quietly degrade performance.
Learning Objective: align operations to data structure strengths.
Core Theory: Dictionary expected O(1) lookup, List O(1) index access and O(n) scan, HashSet for fast uniqueness checks.
Complexity / Trade-offs:
- List: ordered, index-friendly, slower membership checks
- Dictionary: key lookups and updates, no inherent ordering
- HashSet: fast uniqueness and membership
Practice:
- replace duplicate-check loop with HashSet strategy`
            },
            {
                title: "Lesson 2: Generics and Constraints",
                content: `Why this matters: generics improve reuse without unsafe casting.
Learning Objective: write constrained generic methods and classes.
Core Theory: constraints (where T : class/new()/interface) express capabilities and prevent invalid use.
Practice:
- implement generic repository interface with constraints`
            },
            {
                title: "Lesson 3: LINQ Fundamentals and Deferred Execution",
                content: `Why this matters: LINQ can improve readability or hide expensive behavior depending on usage.
Learning Objective: compose pipelines and understand execution timing.
Core Theory: many LINQ operations are deferred until enumeration; materialization via ToList/ToArray changes behavior.
Diagram (Mermaid):
flowchart LR
  A[source] --> B[Where]
  B --> C[Select]
  C --> D[ToList]
Practice:
- detect and fix accidental multiple enumeration`,
                code: `var topNames = students
    .Where(s => s.Score >= 85)
    .OrderByDescending(s => s.Score)
    .Select(s => s.Name)
    .ToList();`
            },
            {
                title: "Lesson 4: Exception Strategy and Domain Errors",
                content: `Why this matters: broad exception handling reduces observability and correctness.
Learning Objective: define exception boundaries and map domain failures explicitly.
Core Theory: catch specific exceptions, enrich with context, and preserve stack where needed.
Practice:
- replace catch(Exception) with targeted exception handling`,
                code: `public sealed class InvalidScoreException : Exception
{
    public InvalidScoreException(string message) : base(message) { }
}`
            },
            {
                title: "Lesson 5: Pipeline Design with LINQ plus Validation",
                content: `Why this matters: real data pipelines include bad records and partial success.
Learning Objective: design parse -> validate -> transform -> summarize stages.
Core Theory: isolate validation rules and maintain reject buckets with reason codes.
Practice:
- produce summary report and reject report in one processing run`
            },
            {
                title: "Mini Project: Student Record Manager v2",
                content: `Project Goal: build a robust student record system using LINQ analytics and explicit exceptions.
Required Features:
- add/update/query students by id and course
- compute top performers and score distribution
- track invalid operations with domain exceptions
Evaluation Signals:
- correct collection choices
- readable LINQ queries
- predictable failure behavior`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
- add import mode with malformed-row reject logging
- support custom comparer for case-insensitive ids`
            },
            {
                title: "Module Quiz",
                content: `1) Deferred execution means: A) query runs immediately B) query runs on enumeration C) query never runs D) compile only
2) Best for fast key lookup: A) List B) Dictionary C) Queue D) Array
3) Generic constraints are used to: A) slow runtime B) enforce API capabilities C) remove typing D) avoid classes
4) Recommended catch style: A) catch everything silently B) catch specific and handle contextually C) never catch D) swallow errors
5) HashSet is strongest for: A) ordered indexing B) uniqueness checks C) range sorting D) file IO`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
- deferred execution pitfalls
- List vs Dictionary trade-offs
- generic constraints use-cases
- exception strategy design`
            },
            {
                title: "Module Summary",
                content: `You can now build data-heavy C# workflows with strong collection choices, cleaner LINQ pipelines, and explicit error contracts.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module focuses on asynchronous programming, concurrency safety, cancellation, and resilient background execution.`
            }
        ]
    },
    {
        number: "Module 4",
        title: "Async, Concurrency, and Resilience",
        description: "Master async/await, task composition, cancellation, and concurrency-safe patterns used in scalable .NET services.",
        duration: "100 min",
        lessons: "14 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Task and async/await", "Task.WhenAll/WhenAny", "CancellationToken", "ConfigureAwait Basics", "IAsyncEnumerable", "Thread Safety", "Locks and Concurrent Collections", "Background Services", "Retry and Timeout Patterns", "Transient Fault Handling", "Resilience Trade-offs", "Async Exception Flow", "Diagnostics", "Performance Considerations"],
        detailedDescription: "Advanced-intermediate module for building robust asynchronous systems without hidden concurrency bugs.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 4
Module Name: Async, Concurrency, and Resilience
Difficulty: Intermediate to Advanced
Estimated Reading Time: 100 min
Estimated Completion Time: 10-11 hours
Prerequisites: Modules 1-3
Learning Objectives:
- reason correctly about async control flow and scheduling
- implement cancellation and timeout behavior safely
- build resilience for transient dependency failures
Skills Gained:
- production async design
- concurrency safety awareness
- resilient integration patterns`
            },
            {
                title: "Lesson 1: async/await Execution Model",
                content: `Why this matters: async misunderstandings cause deadlocks, hangs, and poor throughput.
Learning Objective: model await suspension and continuation behavior.
Core Theory: await may suspend execution, release thread, and resume later on continuation context.
Diagram (Mermaid):
sequenceDiagram
  participant Caller
  participant Method as Async Method
  participant IO as I/O Task
  Caller->>Method: call
  Method->>IO: await request
  IO-->>Method: completes later
  Method-->>Caller: continuation result
Practice:
- refactor blocking .Result code to async/await`,
                code: `public static async Task<string> FetchAsync(HttpClient client, string url, CancellationToken ct)
{
    using var response = await client.GetAsync(url, ct);
    response.EnsureSuccessStatusCode();
    return await response.Content.ReadAsStringAsync(ct);
}`
            },
            {
                title: "Lesson 2: Cancellation and Timeouts",
                content: `Why this matters: long-running operations must be interruptible.
Learning Objective: propagate CancellationToken across boundaries.
Core Theory: cancellation is cooperative; each layer must honor token.
Practice:
- add cancellation flow through controller -> service -> repository`
            },
            {
                title: "Lesson 3: Concurrency Safety with Shared State",
                content: `Why this matters: race conditions create non-deterministic bugs.
Learning Objective: protect shared state with lock discipline or concurrent collections.
Core Theory: minimize shared mutable state; prefer immutable messages when possible.
Practice:
- replace shared List with ConcurrentDictionary in worker logic`
            },
            {
                title: "Lesson 4: Resilience Patterns",
                content: `Why this matters: APIs and networks fail intermittently in production.
Learning Objective: apply bounded retry and timeout policies responsibly.
Core Theory: retries should target transient failures and use backoff to avoid overload.
Practice:
- implement retry with max attempts and jittered delay`
            },
            {
                title: "Lesson 5: Async Streams and Background Processing",
                content: `Why this matters: streaming and background jobs are common in modern services.
Learning Objective: use IAsyncEnumerable for incremental processing.
Core Theory: async streams allow progressive consumption without full materialization.
Practice:
- process event feed with await foreach and cancellation support`
            },
            {
                title: "Mini Project: Concurrent File Ingestion Worker",
                content: `Project Goal: build a worker that ingests files concurrently with cancellation and retry safety.
Required Features:
- async read and parse pipeline
- cancellation-aware processing
- bounded retry for transient failures
- deterministic summary output
Evaluation Signals:
- no deadlocks or blocking waits
- clear failure categorization
- predictable cancellation behavior`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
- add per-file timeout and dead-letter output
- expose metrics summary for success/failure/timeout counts`
            },
            {
                title: "Module Quiz",
                content: `1) await primarily does: A) blocks thread always B) may suspend and resume later C) compiles to sync code D) avoids exceptions
2) CancellationToken is: A) forced kill switch B) cooperative cancellation signal C) logging helper D) serializer
3) Common async anti-pattern: A) await Task.Delay B) using .Result in async path C) Task.WhenAll D) cancellation checks
4) Retry policy should be: A) infinite by default B) bounded and selective C) random always D) absent
5) IAsyncEnumerable helps with: A) compile speed B) streaming async data C) class inheritance D) DI setup`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
- async deadlock scenarios
- cancellation propagation strategy
- thread safety design choices
- retry/timeout trade-offs`
            },
            {
                title: "Module Summary",
                content: `You can now build asynchronous C# workflows that are resilient, cancellation-aware, and safer under concurrent load.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module moves into ASP.NET Core APIs, dependency injection, EF Core basics, and production architecture patterns.`
            }
        ]
    },
    {
        number: "Module 5",
        title: "ASP.NET Core, Data Access, and Architecture",
        description: "Build production-style backend services with ASP.NET Core, DI, validation, EF Core basics, and layered architecture.",
        duration: "105 min",
        lessons: "14 lessons",
        isNew: false,
        isLocked: false,
        topics: ["ASP.NET Core Pipeline", "Controllers and Minimal APIs", "Dependency Injection", "DTOs and Validation", "Middleware", "Configuration", "Logging", "EF Core Basics", "Migrations", "Repository Pattern", "Service Layer", "Caching Basics", "Authentication Overview", "API Versioning Basics"],
        detailedDescription: "Production-focused module that connects C# language skills to real backend service design and operations.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 5
Module Name: ASP.NET Core, Data Access, and Architecture
Difficulty: Intermediate to Advanced
Estimated Reading Time: 105 min
Estimated Completion Time: 11-12 hours
Prerequisites: Modules 1-4
Learning Objectives:
- design API endpoints with clear contracts and validation
- implement layered architecture with DI and data access abstraction
- add observability and configuration for deployable services
Skills Gained:
- backend API engineering
- service and repository boundary design
- production readiness practices`
            },
            {
                title: "Lesson 1: Request Pipeline and Middleware",
                content: `Why this matters: middleware order controls cross-cutting behavior and security.
Learning Objective: reason about request flow through ASP.NET Core pipeline.
Core Theory: each middleware can inspect, short-circuit, or pass request to next component.
Diagram (Mermaid):
flowchart LR
  A[Request] --> B[Auth Middleware]
  B --> C[Validation Middleware]
  C --> D[Controller]
  D --> E[Response]
Practice:
- add timing middleware and verify ordering impact`
            },
            {
                title: "Lesson 2: Controllers, DTOs, and Validation",
                content: `Why this matters: API contracts define integration reliability.
Learning Objective: use request/response DTOs and model validation.
Core Theory: DTOs decouple API shape from domain model; validation should fail fast with clear responses.
Practice:
- create POST endpoint with validation and typed response`
            },
            {
                title: "Lesson 3: Dependency Injection and Service Boundaries",
                content: `Why this matters: DI reduces coupling and improves testability.
Learning Objective: register and consume services through interfaces.
Core Theory: service lifetime (singleton/scoped/transient) affects behavior and safety.
Practice:
- identify a lifetime bug caused by wrong registration`
            },
            {
                title: "Lesson 4: EF Core Basics and Repository Abstraction",
                content: `Why this matters: data access should be maintainable and test-friendly.
Learning Objective: perform CRUD with EF Core while preserving domain boundaries.
Core Theory: DbContext tracks entities; migrations evolve schema over time.
Practice:
- add migration and repository methods for one aggregate`
            },
            {
                title: "Lesson 5: Logging, Config, and Deployment Readiness",
                content: `Why this matters: production support depends on diagnostics and predictable configuration.
Learning Objective: implement structured logs and environment-specific settings.
Core Theory: centralize configuration sources and avoid hardcoded secrets.
Practice:
- add environment-specific appsettings and startup validation`
            },
            {
                title: "Mini Project: Task Tracker API",
                content: `Project Goal: build a layered Task Tracker API with validation, DI, and persistence.
Required Features:
- create/update/list task endpoints
- DTO validation and standardized error responses
- service and repository layers
- EF Core persistence with migration
Evaluation Signals:
- clean API contracts
- architecture boundary clarity
- reliable runtime behavior`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
- add paging and filtering to list endpoint
- include response metadata for total count and page info`
            },
            {
                title: "Module Quiz",
                content: `1) Middleware order in ASP.NET Core is: A) irrelevant B) critical C) random D) compile-time only
2) DTOs primarily help with: A) thread pool size B) contract isolation C) migration speed D) GC tuning
3) Scoped service lifetime is typically per: A) application lifetime B) request C) class file D) assembly
4) EF Core migration is used for: A) caching B) schema evolution C) logging D) encryption
5) Config best practice: A) hardcode secrets B) environment-driven configuration C) skip validation D) single file only`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
- middleware flow explanation
- DTO vs domain model distinction
- DI lifetime trade-offs
- EF Core and repository pattern decisions`
            },
            {
                title: "Module Summary",
                content: `You can now build production-style ASP.NET Core APIs with maintainable architecture, data access patterns, and operational readiness.`
            },
            {
                title: "Next Module Bridge",
                content: `Final module integrates testing, architecture review, and full capstone delivery for portfolio and interview readiness.`
            }
        ]
    },
    {
        number: "Module 6",
        title: "Production C# Capstone and Interview Readiness",
        description: "Deliver a complete C# backend project with testing, architecture rationale, operational checks, and interview-grade documentation.",
        duration: "120 min",
        lessons: "15 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Scope Definition", "Architecture Blueprint", "Domain Model", "API Contracts", "Persistence", "Error Strategy", "Observability", "Automated Testing", "Integration Tests", "Performance Baseline", "Security Checklist", "Documentation", "ADR Basics", "Final Assessment", "Demo Walkthrough"],
        detailedDescription: "Capstone module that consolidates C# language, backend architecture, and engineering communication into one delivery path.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 6
Module Name: Production C# Capstone and Interview Readiness
Difficulty: Advanced
Estimated Reading Time: 120 min
Estimated Completion Time: 12-14 hours
Prerequisites: Modules 1-5
Learning Objectives:
- deliver an end-to-end C# backend application
- enforce quality through tests and operational diagnostics
- explain design choices and trade-offs in interview settings
Skills Gained:
- full delivery lifecycle ownership
- architecture communication confidence
- portfolio-grade implementation quality`
            },
            {
                title: "Lesson 1: Scope, Constraints, and MVP Planning",
                content: `Why this matters: unmanaged scope causes incomplete and unstable projects.
Learning Objective: define MVP features and non-functional boundaries.
Core Theory: prioritize core workflows first, defer optional enhancements.
Practice:
- write feature breakdown with acceptance criteria`
            },
            {
                title: "Lesson 2: Architecture and Boundary Enforcement",
                content: `Why this matters: capstone quality is judged by maintainability, not only feature count.
Learning Objective: enforce clean separation across API, service, domain, and infrastructure.
Core Theory: each layer should have one reason to change.
Practice:
- perform architecture review and list coupling hotspots`
            },
            {
                title: "Lesson 3: Test Strategy and Coverage Priorities",
                content: `Why this matters: tests enable safe iteration and bug prevention.
Learning Objective: build a practical test matrix for unit and integration tests.
Core Theory: prioritize business-critical paths and failure scenarios.
Practice:
- implement tests for happy path, validation failure, and dependency failure`,
                code: `public sealed class TaskServiceTests
{
    [Fact]
    public async Task CreateTask_ShouldReject_EmptyTitle()
    {
        var repo = new InMemoryTaskRepository();
        var service = new TaskService(repo);

        await Assert.ThrowsAsync<ValidationException>(() => service.CreateAsync(""));
    }
}`
            },
            {
                title: "Lesson 4: Operational Readiness Checklist",
                content: `Why this matters: production deployment requires more than passing tests.
Learning Objective: validate logging, configuration, health checks, and error responses.
Core Theory: use structured logs, correlation ids, and clear health endpoints.
Practice:
- add startup checks for required configuration values`
            },
            {
                title: "Lesson 5: Documentation and Architecture Narrative",
                content: `Why this matters: interviewers evaluate your reasoning and communication, not just source files.
Learning Objective: produce README, architecture notes, and demo flow.
Core Theory: document context, decision, alternatives, and consequences for key architectural choices.
Practice:
- write one ADR for storage strategy decision`
            },
            {
                title: "Capstone Project Options",
                content: `Choose one capstone:
- Task Workflow API
- Inventory Operations API
- Student Lifecycle API
- Personal Finance API
Minimum Delivery:
- layered architecture
- validation and exception strategy
- persistence and migration
- tests, logs, and docs`
            },
            {
                title: "Final Assessment",
                content: `Assessment Format:
- concept check across modules 1-6
- implementation task with constraints
- debugging and failing-test fix exercise
- architecture trade-off discussion
Pass Criteria:
- correctness and maintainability
- test-backed behavior
- clarity of design explanations`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
- add one advanced feature (caching, authorization rule, or analytics endpoint)
- preserve architecture boundaries and test coverage`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
- explain solution architecture and dependency flow
- defend key trade-offs
- show debugging and test strategy
- deliver concise end-to-end demo narrative`
            },
            {
                title: "Module Summary",
                content: `You now have a full production-style C# capstone path with implementation depth, quality practices, and interview readiness.`
            },
            {
                title: "Course Completion Path",
                content: `Complete final assessment, polish capstone documentation, and add your demo walkthrough to your project portfolio.`
            }
        ]
    }
];
courseData.cppProgramming = [
    {
        number: "Module 1",
        title: "C++ Foundations and Compilation Model",
        description: "Build strong C++ fundamentals with compilation pipeline, type system basics, control flow, and practical debugging discipline.",
        duration: "90 min",
        lessons: "14 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Introduction and History", "Compilation and Linking", "Toolchain Basics", "Types and Initialization", "Const Correctness", "References", "Functions", "Control Flow", "Input/Output", "Namespaces", "Header and Source Separation", "Error Messages", "Build Configurations", "Assertions", "Debugging Basics"],
        detailedDescription: "Beginner-first module that explains how C++ programs are built and executed, not just how syntax looks.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 1
Module Name: C++ Foundations and Compilation Model
Difficulty: Beginner
Estimated Reading Time: 90 min
Estimated Completion Time: 8-9 hours
Prerequisites: None
Learning Objectives:
- explain preprocessing, compilation, linking, and execution flow
- write type-safe C++ basics using clear initialization patterns
- diagnose compile and runtime errors with structured debugging
Skills Gained:
- C++ toolchain literacy
- syntax and type-system confidence
- practical debugging workflow`
            },
                        {
                                title: "Lesson 0: C++ Introduction and History",
                                content: `Why this matters: C++ history clarifies why the language prioritizes performance, control, and zero-cost abstractions.
Learning Objective: understand C++ origins and modern evolution.
Core Theory: C++ grew from C with object-oriented and generic programming support, then evolved with modern standards (C++11 onward) that improved safety and expressiveness while preserving performance.
Diagram (Mermaid):
timeline
    1985 : First commercial C++ release
    1998 : First ISO C++ standard
    2011 : C++11 modern era begins
    2017 : C++17 mainstream modern features
    2020 : C++20 concepts and ranges
Common Mistakes: learning only legacy C-style patterns and missing modern C++ safety tools.
Recap:
- C++ emphasizes performance with explicit control
- Modern standards improved safety without sacrificing speed
- Industry use remains strong in systems, engines, finance, and embedded
Practice:
- compare one pre-modern and modern C++ approach for resource management`
                        },
            {
                title: "Lesson 1: How C++ Code Becomes an Executable",
                content: `Why this matters: most beginner friction in C++ starts at build stage, not runtime logic.
Learning Objective: understand preprocessing, compilation, and linking.
Core Theory: source files are preprocessed, compiled to object files, then linked into an executable.
Diagram (Mermaid):
flowchart LR
  A[.cpp/.h files] --> B[preprocessor]
  B --> C[compiler]
  C --> D[object files]
  D --> E[linker]
  E --> F[executable]
Worked Example: split one program across two translation units and link successfully.
Common Mistakes: missing definitions, duplicate symbols, header include cycles.
Recap:
- compiler checks each translation unit
- linker resolves cross-file symbols
- build errors require stage-aware troubleshooting
Practice:
- create two-file project and fix one linker error`,
                code: `#include <iostream>

int main() {
    std::cout << "Hello from C++ toolchain" << std::endl;
    return 0;
}`
            },
            {
                title: "Lesson 2: Types, Initialization, and Const Correctness",
                content: `Why this matters: undefined behavior and accidental mutation often begin with weak initialization habits.
Learning Objective: use direct/list initialization and const constraints intentionally.
Core Theory: brace initialization avoids narrowing; const expresses immutability at API boundaries.
Common Mistakes: uninitialized variables and mutable parameters that should be const.
Recap:
- initialize values at declaration
- prefer const for read-only intent
- enforce invariants through function signatures`
            },
            {
                title: "Lesson 3: Functions, References, and Parameter Passing",
                content: `Why this matters: C++ performance and correctness depend heavily on function signature design.
Learning Objective: choose pass-by-value, const reference, and mutable reference appropriately.
Core Theory: pass-by-value copies; const reference avoids copy while protecting input; non-const reference enables mutation.
Practice:
- refactor heavy-copy function to const reference parameters`,
                code: `int sum(const std::vector<int>& values) {
    int total = 0;
    for (int v : values) total += v;
    return total;
}`
            },
            {
                title: "Lesson 4: Header/Source Organization and Namespaces",
                content: `Why this matters: scalable C++ codebases require disciplined boundaries across files.
Learning Objective: separate declarations and definitions cleanly.
Core Theory: headers expose interfaces; source files hold implementation; namespaces prevent symbol collisions.
Practice:
- move utility functions into namespaced header/source pair`
            },
            {
                title: "Lesson 5: Compile Errors vs Runtime Errors",
                content: `Why this matters: different failure classes require different debugging strategies.
Learning Objective: read compiler diagnostics and runtime traces systematically.
Core Theory: compiler errors often cascade; fix earliest relevant diagnostic first.
Practice:
- resolve one syntax error, one type mismatch, and one runtime crash in sequence`
            },
            {
                title: "Mini Project: Student Result Console",
                content: `Project Goal: build a robust console app for student score analytics.
Required Features:
- read student records and validate values
- compute grade distribution and class summary
- separate declarations and definitions into multiple files
Evaluation Signals:
- clean compile with warnings addressed
- predictable validation behavior
- readable modular structure`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
- add CLI flags for input file path and output mode
- keep behavior deterministic for invalid input handling`
            },
            {
                title: "Module Quiz",
                content: `1) Linking primarily does: A) syntax checking B) symbol resolution across object files C) memory allocation D) runtime scheduling
2) Header files should mainly contain: A) random globals B) declarations/interfaces C) only main() D) linker scripts
3) const reference is best for: A) mutable output B) large read-only input C) temporary counters D) preprocessor macros
4) First step when many compile errors appear: A) fix random one B) fix earliest root diagnostic C) disable warnings D) rewrite project
5) Namespace purpose: A) optimize loops B) avoid name collisions C) allocate heap D) open files`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
- compilation vs linking differences
- pass-by-value vs const reference trade-offs
- header/source organization best practices
- interpreting compiler diagnostics`
            },
            {
                title: "Module Summary",
                content: `You can now build and debug foundational C++ programs with stronger toolchain awareness and safer type/initialization habits.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module dives into pointers, references, ownership, and lifetime management where C++ differs most from managed languages.`
            }
        ]
    },
    {
        number: "Module 2",
        title: "Pointers, Ownership, and Lifetime",
        description: "Master pointer semantics, dynamic memory risks, RAII fundamentals, and ownership modeling for safe C++ systems.",
        duration: "100 min",
        lessons: "14 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Pointers and Addresses", "References vs Pointers", "Stack vs Heap", "new/delete", "Dangling Pointers", "Memory Leaks", "RAII Intro", "Smart Pointer Basics", "Move Semantics Intro", "Copy vs Move", "Lifetime Boundaries", "Resource Ownership", "Destructors", "Rule of Zero"],
        detailedDescription: "Core C++ memory module focused on safety, ownership clarity, and lifetime correctness.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 2
Module Name: Pointers, Ownership, and Lifetime
Difficulty: Intermediate
Estimated Reading Time: 100 min
Estimated Completion Time: 9-10 hours
Prerequisites: Module 1
Learning Objectives:
- reason about memory lifetime and ownership explicitly
- avoid common pointer-related bugs and undefined behavior
- apply RAII and modern ownership patterns
Skills Gained:
- memory safety mindset
- deterministic resource management
- stronger API ownership contracts`
            },
            {
                title: "Lesson 1: Pointers, References, and Lifetime Boundaries",
                content: `Why this matters: pointer misuse leads to crashes and security vulnerabilities.
Learning Objective: distinguish aliasing, ownership, and borrowing roles.
Core Theory: raw pointers may be nullable and rebindable; references are non-null aliases once bound.
Diagram (Mermaid):
flowchart TD
  A[object lifetime] --> B[owning handle]
  B --> C[borrowed reference]
  B --> D[observing pointer]
Practice:
- annotate function signatures with ownership intent`
            },
            {
                title: "Lesson 2: Stack vs Heap and Allocation Costs",
                content: `Why this matters: allocation strategy affects both performance and reliability.
Learning Objective: choose automatic vs dynamic storage appropriately.
Core Theory: stack allocation is automatic and fast; heap allocation is flexible but requires ownership discipline.
Common Mistakes: unnecessary heap allocations for short-lived objects.
Practice:
- convert heap-based local object usage to stack allocation where safe`
            },
            {
                title: "Lesson 3: RAII and Destructor-Based Cleanup",
                content: `Why this matters: manual cleanup paths are fragile under exceptions.
Learning Objective: tie resource release to object lifetime.
Core Theory: RAII ensures cleanup in destructor regardless of normal return or throw.
Practice:
- wrap file/resource handle in RAII class`
            },
            {
                title: "Lesson 4: Smart Pointers and Ownership Models",
                content: `Why this matters: modern C++ prefers explicit ownership semantics.
Learning Objective: choose unique_ptr and shared_ptr intentionally.
Core Theory: unique_ptr models single ownership; shared_ptr models shared lifetime with reference counting overhead.
Practice:
- refactor new/delete code to unique_ptr`,
                code: `#include <memory>

struct User {
    int id;
};

int main() {
    auto user = std::make_unique<User>();
    user->id = 42;
    return 0;
}`
            },
            {
                title: "Lesson 5: Copy, Move, and Rule of Zero",
                content: `Why this matters: accidental copying can be expensive and semantically wrong.
Learning Objective: understand move semantics and prefer Rule of Zero where possible.
Core Theory: classes owning resources may need custom move/copy behavior, but standard containers/smart pointers often remove that need.
Practice:
- identify a class that can follow Rule of Zero`
            },
            {
                title: "Mini Project: Memory-Safe Resource Tracker",
                content: `Project Goal: build a small tracker app that manages resource objects with explicit ownership and safe cleanup.
Required Features:
- create/update/list resources
- use unique_ptr for owned entities
- avoid raw owning pointers and manual delete
Evaluation Signals:
- no leak-prone ownership patterns
- clear lifetime boundaries
- exception-safe cleanup`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
- add shared observer views without breaking ownership model
- justify any use of shared_ptr in design notes`
            },
            {
                title: "Module Quiz",
                content: `1) unique_ptr represents: A) shared ownership B) single ownership C) no ownership D) compile optimization
2) RAII primarily guarantees: A) faster compile B) deterministic cleanup C) fewer headers D) no exceptions
3) Dangling pointer means: A) points to live object B) points to invalid lifetime C) is always null D) is const
4) Preferred modern owning pointer type: A) raw pointer B) unique_ptr C) reference D) void*
5) Rule of Zero encourages: A) more manual memory code B) relying on safe standard types C) no constructors D) no classes`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
- unique_ptr vs shared_ptr trade-offs
- RAII explanation with practical example
- dangling pointer scenarios
- move semantics intuition`
            },
            {
                title: "Module Summary",
                content: `You can now design C++ code with explicit ownership, safer lifetimes, and modern RAII-driven resource management.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module introduces STL containers, iterators, algorithms, and complexity-aware choices for data-heavy workloads.`
            }
        ]
    },
    {
        number: "Module 3",
        title: "STL, Algorithms, and Generic Programming",
        description: "Use STL containers and algorithms effectively with iterators, complexity awareness, and template-based reuse.",
        duration: "105 min",
        lessons: "14 lessons",
        isNew: false,
        isLocked: false,
        topics: ["vector", "deque", "list", "map and unordered_map", "set and unordered_set", "Iterators", "STL Algorithms", "Custom Comparators", "Lambda Functions", "Template Functions", "Template Classes", "Concepts Basics", "Complexity Trade-offs", "Allocator Awareness"],
        detailedDescription: "Intermediate module for high-signal STL usage and performance-informed generic design.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 3
Module Name: STL, Algorithms, and Generic Programming
Difficulty: Intermediate
Estimated Reading Time: 105 min
Estimated Completion Time: 10-11 hours
Prerequisites: Modules 1-2
Learning Objectives:
- choose containers by operation and ordering needs
- use STL algorithms with iterators and lambdas
- write reusable templates with clear constraints
Skills Gained:
- container and complexity reasoning
- expressive standard algorithm usage
- generic programming foundations`
            },
            {
                title: "Lesson 1: Container Selection by Workload",
                content: `Why this matters: performance regressions often come from mismatched data structures.
Learning Objective: map operation patterns to container choices.
Core Theory: vector excels at contiguous storage and iteration; unordered_map offers expected O(1) lookup; map provides ordered O(log n) operations.
Complexity / Trade-offs:
- vector: fast traversal, expensive middle insert/erase
- unordered_map: fast expected lookup, unstable order
- map: ordered iteration with tree-based overhead
Practice:
- replace one linear search workflow with unordered_map index`
            },
            {
                title: "Lesson 2: Iterators, Algorithms, and Lambdas",
                content: `Why this matters: STL algorithms reduce boilerplate and improve clarity when used correctly.
Learning Objective: compose find/transform/sort pipelines.
Core Theory: algorithms operate on iterator ranges; lambdas customize behavior locally.
Diagram (Mermaid):
flowchart LR
  A[container begin/end] --> B[std::transform]
  B --> C[std::sort]
  C --> D[result range]
Practice:
- rewrite loop-based transform with std::transform`,
                code: `std::vector<int> values{5, 2, 8, 1};
std::sort(values.begin(), values.end());

std::vector<int> doubled(values.size());
std::transform(values.begin(), values.end(), doubled.begin(),
               [](int v) { return v * 2; });`
            },
            {
                title: "Lesson 3: Template Reuse and Constraints",
                content: `Why this matters: templates enable zero-cost abstraction but can become hard to read without constraints.
Learning Objective: write focused generic functions and constrain assumptions.
Core Theory: templates are compile-time polymorphism; concepts (modern C++) express intent and improve diagnostics.
Practice:
- add a concept-constrained utility template`
            },
            {
                title: "Lesson 4: Comparator Design and Ordering Semantics",
                content: `Why this matters: subtle comparator bugs break sorting and set/map correctness.
Learning Objective: implement strict weak ordering comparators.
Core Theory: comparator must be consistent and transitive.
Practice:
- implement custom comparator for multi-field record sorting`
            },
            {
                title: "Lesson 5: Algorithmic Thinking with STL",
                content: `Why this matters: STL is most powerful when used with algorithmic intent.
Learning Objective: combine containers and algorithms for clear pipelines.
Core Theory: prefer standard algorithms before writing manual loops unless profiling proves need.
Practice:
- build top-N summary report using partial_sort and map accumulators`
            },
            {
                title: "Mini Project: Inventory Analytics Engine",
                content: `Project Goal: build analytics over inventory records using STL containers and algorithms.
Required Features:
- index items by id and category
- compute low-stock and fast-moving summaries
- output sorted reports by configurable criteria
Evaluation Signals:
- correct container usage
- clear algorithm pipelines
- performance-aware design notes`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
- add templated report utility reusable across record types
- include complexity notes for each major operation`
            },
            {
                title: "Module Quiz",
                content: `1) map typically provides: A) expected O(1) lookup B) ordered O(log n) operations C) constant insertion at front D) fixed capacity
2) unordered_map is best when: A) sorted iteration needed B) fast expected key lookup needed C) contiguous memory is mandatory D) random pointers required
3) STL algorithms mainly operate on: A) classes only B) iterator ranges C) namespaces D) macros
4) Template constraints improve: A) runtime speed only B) diagnostics and contract clarity C) linker behavior D) memory leaks
5) Comparator must satisfy: A) random ordering B) strict weak ordering C) hash collision D) reference counting`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
- vector vs list trade-offs
- map vs unordered_map decision criteria
- lambda and algorithm use-cases
- template constraints and readability`
            },
            {
                title: "Module Summary",
                content: `You can now design C++ data workflows with STL-first thinking, complexity awareness, and reusable generic abstractions.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module moves into modern C++ features including move semantics depth, RAII patterns, and advanced language constructs.`
            }
        ]
    },
    {
        number: "Module 4",
        title: "Modern C++: RAII, Move Semantics, and Language Features",
        description: "Apply modern C++ features such as RAII, move semantics, smart pointers, and safer expressive language constructs.",
        duration: "100 min",
        lessons: "14 lessons",
        isNew: false,
        isLocked: false,
        topics: ["RAII Deep Dive", "Move Semantics", "Rule of Five", "Rule of Zero", "Smart Pointer Patterns", "constexpr", "auto and Type Deduction", "Structured Bindings", "std::optional", "std::variant", "std::span Basics", "Error Handling Patterns", "C++17/C++20 Highlights", "API Design in Modern C++"],
        detailedDescription: "Advanced-intermediate module focused on writing expressive and safer modern C++ code.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 4
Module Name: Modern C++: RAII, Move Semantics, and Language Features
Difficulty: Intermediate to Advanced
Estimated Reading Time: 100 min
Estimated Completion Time: 10-11 hours
Prerequisites: Modules 1-3
Learning Objectives:
- use move semantics to avoid unnecessary copies
- apply modern utility types for safer APIs
- design maintainable modern C++ interfaces
Skills Gained:
- modern language fluency
- safer ownership and transfer patterns
- expressive API and type design`
            },
            {
                title: "Lesson 1: Move Semantics in Practice",
                content: `Why this matters: unnecessary copies can dominate performance in large data paths.
Learning Objective: understand lvalues/rvalues and move operations.
Core Theory: move transfers resources from temporary or explicitly moved objects.
Practice:
- profile copy-heavy code path and reduce copies with move-aware design`,
                code: `std::vector<int> buildData() {
    std::vector<int> v(1000, 1);
    return v; // NRVO/move
}`
            },
            {
                title: "Lesson 2: Rule of Five vs Rule of Zero",
                content: `Why this matters: custom resource ownership requires explicit copy/move behavior.
Learning Objective: decide when to implement special member functions.
Core Theory: if class manages resource directly, Rule of Five may apply; otherwise prefer Rule of Zero.
Practice:
- audit a class for unnecessary custom copy/move code`
            },
            {
                title: "Lesson 3: optional, variant, and Safer Return Contracts",
                content: `Why this matters: expressive types reduce invalid states and ambiguous error codes.
Learning Objective: replace sentinel values with explicit modeling.
Core Theory: optional models presence/absence; variant models closed alternatives.
Practice:
- refactor parse result from bool+out to variant/optional`
            },
            {
                title: "Lesson 4: constexpr and Compile-Time Computation",
                content: `Why this matters: compile-time evaluation can improve correctness and performance for static logic.
Learning Objective: identify suitable constexpr use-cases.
Core Theory: constexpr functions can run at compile time when given constant expressions.
Practice:
- convert one utility to constexpr and verify compile-time usage`
            },
            {
                title: "Lesson 5: Modern API Design Guidelines",
                content: `Why this matters: strong APIs prevent misuse and lower maintenance cost.
Learning Objective: design interfaces with clear ownership and minimal surprises.
Core Theory: prefer explicit constructors, const correctness, and non-owning views where appropriate.
Practice:
- redesign one legacy API with modern type contracts`
            },
            {
                title: "Mini Project: Modernized Report Processor",
                content: `Project Goal: modernize a legacy report module using move semantics and modern utility types.
Required Features:
- replace raw ownership with smart pointers or value semantics
- use optional/variant for result modeling
- eliminate unnecessary copies in hot paths
Evaluation Signals:
- improved readability and safety
- fewer ownership ambiguities
- measurable copy reduction`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
- add compile-time validated config constants with constexpr
- provide benchmark notes before and after modernization`
            },
            {
                title: "Module Quiz",
                content: `1) Move semantics primarily optimize: A) syntax coloring B) resource transfer efficiency C) linker size D) recursion depth
2) Rule of Zero suggests: A) always implement destructor B) rely on safe member types and defaults C) use raw pointers only D) avoid classes
3) optional is best for: A) guaranteed result B) possibly missing value C) threading D) macros
4) variant models: A) dynamic array B) closed set of alternative types C) file streams D) inheritance only
5) constexpr allows: A) runtime-only evaluation B) potential compile-time evaluation C) memory leak detection D) auto-linking`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
- lvalue/rvalue and move semantics intuition
- Rule of Five vs Rule of Zero
- optional/variant design decisions
- modern C++ API best practices`
            },
            {
                title: "Module Summary",
                content: `You can now apply modern C++ constructs to build safer, clearer, and more efficient APIs and implementations.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module covers concurrency, synchronization, profiling, and performance engineering in systems-scale C++ code.`
            }
        ]
    },
    {
        number: "Module 5",
        title: "Concurrency and Performance Engineering in C++",
        description: "Build concurrent C++ systems safely and optimize with profiling-driven decisions instead of guesswork.",
        duration: "105 min",
        lessons: "14 lessons",
        isNew: false,
        isLocked: false,
        topics: ["std::thread", "Mutex and Locking", "Deadlocks", "Condition Variables", "Atomics Basics", "Memory Ordering Intro", "Task-Based Concurrency", "Thread Pools Concept", "Profiling Workflow", "Cache and Allocation Costs", "Benchmark Pitfalls", "False Sharing", "Performance Trade-offs", "Optimization Discipline"],
        detailedDescription: "Advanced module that combines concurrent correctness with measured performance tuning.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 5
Module Name: Concurrency and Performance Engineering in C++
Difficulty: Advanced
Estimated Reading Time: 105 min
Estimated Completion Time: 11-12 hours
Prerequisites: Modules 1-4
Learning Objectives:
- implement thread-safe workflows with explicit synchronization
- avoid common concurrency hazards and contention bottlenecks
- optimize based on profiler evidence and benchmark hygiene
Skills Gained:
- concurrent systems reasoning
- synchronization safety practices
- performance investigation discipline`
            },
            {
                title: "Lesson 1: Threading Basics and Shared-State Hazards",
                content: `Why this matters: race conditions cause nondeterministic failures that are hard to reproduce.
Learning Objective: reason about shared state and critical sections.
Core Theory: concurrent writes to shared data require synchronization.
Diagram (Mermaid):
sequenceDiagram
  participant T1 as Thread 1
  participant C as Counter
  participant T2 as Thread 2
  T1->>C: read
  T2->>C: read
  T1->>C: write+1
  T2->>C: write+1 (lost update)
Practice:
- fix race condition with mutex`
            },
            {
                title: "Lesson 2: Locks, Deadlocks, and Safe Coordination",
                content: `Why this matters: incorrect lock ordering can freeze production systems.
Learning Objective: design lock strategy that avoids deadlock.
Core Theory: consistent lock ordering and scoped lock helpers reduce risk.
Practice:
- refactor two-lock code path to deadlock-safe order`,
                code: `std::mutex m1, m2;

void safeWork() {
    std::scoped_lock lock(m1, m2);
    // critical section using both resources
}`
            },
            {
                title: "Lesson 3: Atomics and Contention Trade-offs",
                content: `Why this matters: mutex-heavy designs can bottleneck under load.
Learning Objective: identify when atomic operations are sufficient.
Core Theory: atomics are good for simple shared counters/flags; complex invariants often need mutexes.
Practice:
- replace mutex-protected counter with atomic counter and compare throughput`
            },
            {
                title: "Lesson 4: Profiling and Benchmarking Workflow",
                content: `Why this matters: optimization without measurement often worsens code.
Learning Objective: establish baseline, identify hotspot, optimize, and re-measure.
Core Theory: benchmark harnesses need warmup, stable input, and fair comparisons.
Practice:
- capture baseline and optimize one hotspot with clear before/after results`
            },
            {
                title: "Lesson 5: Cache-Aware and Allocation-Aware Design",
                content: `Why this matters: memory access patterns can dominate runtime costs.
Learning Objective: reduce allocations and improve locality where it matters.
Core Theory: contiguous structures and reservation strategies can reduce overhead.
Practice:
- use reserve and data layout improvements in one hot path`
            },
            {
                title: "Mini Project: Concurrent Log Aggregator",
                content: `Project Goal: build a thread-safe log aggregator with measurable performance goals.
Required Features:
- concurrent ingestion from multiple worker threads
- safe aggregation by error code/category
- deterministic final summary output
- benchmark report before and after one optimization
Evaluation Signals:
- thread-safe correctness
- clear synchronization strategy
- evidence-based optimization decisions`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
- add bounded queue with producer/consumer model
- report throughput and latency percentiles on sample workload`
            },
            {
                title: "Module Quiz",
                content: `1) Race condition means: A) compile failure B) order-dependent incorrect behavior C) syntax warning D) linker error
2) scoped_lock helps with: A) random ordering B) coordinated lock acquisition C) file parsing D) template deduction
3) Atomics are best for: A) complex multi-step invariants B) simple counters/flags C) dynamic polymorphism D) header guards
4) First optimization step should be: A) rewrite architecture B) profile baseline C) disable checks D) use macros
5) reserve() on vector helps reduce: A) compile time B) reallocations C) namespaces D) thread creation`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
- diagnosing race conditions
- deadlock prevention strategies
- atomic vs mutex decisions
- profiling-driven optimization examples`
            },
            {
                title: "Module Summary",
                content: `You can now build concurrent C++ workflows with safer synchronization and measurable, profiler-backed performance improvements.`
            },
            {
                title: "Next Module Bridge",
                content: `Final module integrates architecture, quality, and delivery into a portfolio-ready C++ capstone project.`
            }
        ]
    },
    {
        number: "Module 6",
        title: "Production C++ Capstone and Interview Readiness",
        description: "Deliver a full C++ system with architecture, ownership safety, tests, benchmarking, and interview-ready technical narrative.",
        duration: "120 min",
        lessons: "15 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Scope and Requirements", "Architecture Plan", "Data Structures", "Ownership Model", "Error Handling", "CLI/API Boundaries", "Persistence Strategy", "Testing Strategy", "Benchmark Design", "Profiling Results", "Code Review Checklist", "Documentation", "Trade-off Narrative", "Final Assessment", "Demo Walkthrough"],
        detailedDescription: "Capstone module that consolidates systems thinking, C++ engineering rigor, and communication for job readiness.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 6
Module Name: Production C++ Capstone and Interview Readiness
Difficulty: Advanced
Estimated Reading Time: 120 min
Estimated Completion Time: 12-14 hours
Prerequisites: Modules 1-5
Learning Objectives:
- architect and implement a complete C++ project
- enforce safety, correctness, and performance through tests and benchmarks
- communicate design trade-offs clearly for interviews and code reviews
Skills Gained:
- end-to-end C++ delivery discipline
- production quality and performance validation
- technical storytelling confidence`
            },
            {
                title: "Lesson 1: Scope, Constraints, and MVP Planning",
                content: `Why this matters: over-scoped projects fail before quality can emerge.
Learning Objective: define MVP feature boundaries and constraints.
Core Theory: prioritize core workflows and measurable success criteria first.
Practice:
- write acceptance criteria and non-functional goals`
            },
            {
                title: "Lesson 2: Architecture and Ownership Design",
                content: `Why this matters: C++ architecture must explicitly model ownership and lifetime.
Learning Objective: design modules with clear resource boundaries.
Core Theory: separate domain logic, IO adapters, and orchestration layers.
Practice:
- produce architecture diagram with ownership notes`
            },
            {
                title: "Lesson 3: Testing and Reliability Strategy",
                content: `Why this matters: correctness regressions are common in performance-oriented codebases.
Learning Objective: build tests for core behavior and edge-case failures.
Core Theory: combine unit tests for domain logic with integration tests for IO boundaries.
Practice:
- implement tests for success, invalid input, and failure recovery paths`
            },
            {
                title: "Lesson 4: Benchmarking and Optimization Report",
                content: `Why this matters: optimization claims must be evidence-based.
Learning Objective: create reproducible benchmark setup and summarize results.
Core Theory: compare baseline vs optimized versions under same workload.
Practice:
- document one optimization with measured impact`
            },
            {
                title: "Lesson 5: Documentation and Interview Narrative",
                content: `Why this matters: interviewers evaluate engineering judgment, not only source files.
Learning Objective: explain architecture choices, trade-offs, and risk mitigation.
Core Theory: good docs include setup, design rationale, assumptions, and known limits.
Practice:
- prepare 5-minute technical walkthrough script`
            },
            {
                title: "Capstone Project Options",
                content: `Choose one capstone:
- Log Analytics CLI
- Inventory Processing Engine
- Task Scheduling Simulator
- Metrics Aggregation Service
Minimum Delivery:
- modular architecture with ownership clarity
- robust error handling and validation
- tests plus benchmark report
- README with design trade-offs and usage guide`
            },
            {
                title: "Final Assessment",
                content: `Assessment Format:
- concept check across modules 1-6
- implementation task with constraints
- debugging exercise with memory/lifetime issues
- architecture and performance trade-off review
Pass Criteria:
- correctness and safety
- measurable performance reasoning
- clear communication of design decisions`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
- add one advanced feature (parallel processing, caching, or plug-in strategy)
- preserve safety, test stability, and benchmark transparency`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
- ownership and lifetime explanation
- STL and algorithm choices
- concurrency and profiling decisions
- concise end-to-end demo narrative`
            },
            {
                title: "Module Summary",
                content: `You now have a production-grade C++ capstone path with systems-level depth, safety practices, and interview-ready articulation.`
            },
            {
                title: "Course Completion Path",
                content: `Complete final assessment, polish benchmark and documentation artifacts, and prepare your capstone walkthrough for interviews.`
            }
        ]
    }
];
