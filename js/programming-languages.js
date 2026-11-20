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
        description: "Install a JDK, run your first program, then learn values, decisions, loops, methods, and debugging.",
        duration: "80 min",
        lessons: "15 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Setup and First Program", "Variables and Types", "Operators", "Control Flow", "Methods", "Arrays", "Strings", "Enums", "Wrapper Types", "Command Line Build", "Debugging Basics", "Project Structure", "Classpath and Packages", "Optional: Java History and JVM"],
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
                                title: "Optional: Java Introduction and History",
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
                title: "Lesson 1: Install, Compile, and Run Java",
                content: `Goal: see a program you wrote print a message before studying runtime internals.
1. Install a JDK version 21 or later from a trusted distributor. A JDK includes the compiler and runtime; an editor alone does not.
2. Open a new terminal. Run java -version and javac -version. Both should show the intended JDK. If either command is missing, fix your JDK installation/PATH and reopen the terminal.
3. Create an empty learning folder and save the complete example as HelloWorld.java, not HelloWorld.java.txt. The public class name must match the filename.
4. In that folder run javac HelloWorld.java, then java HelloWorld (without .java or .class). Expected output: Hello, Java runtime!
5. Change the message, save, compile again, and run again. Compilation updates the .class file; running alone will not rebuild it.
Read the code: main is the entry point; braces group statements; println prints a line; statements end with semicolons. String[] args receives command-line arguments, which we will use later.
First values and decisions: put int score = 75; inside main, then print score + 5. You should see 80. Add if (score >= 60) { System.out.println("Pass"); } and predict Pass.
First repetition: for (int i = 1; i <= 3; i++) { System.out.println(i); } prints 1, 2, 3 on separate lines.
Diagram (Mermaid):
flowchart LR
  A[Save HelloWorld.java] --> B[javac HelloWorld.java]
  B --> C[java HelloWorld]
  C --> D[Read printed message]
Practice: print your name and the result of 7 + 3 on separate lines. Expected second line: 10.
Later, optional runtime reading: javac produces bytecode. A platform-specific JVM loads and verifies classes, then interprets or JIT-compiles their instructions. javap -c HelloWorld inspects bytecode; recognizing opcodes is not required to complete this module.`,
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
Core Theory: primitives have defined ranges/precision; widening can still lose precision (for example int to float). Narrowing may lose magnitude as well. Integer division truncates toward zero: 7 / 2 is 3; 7 / 2.0 is 3.5. Start with int counts and double measurements; BigDecimal is a later extension for decimal money calculations.
Diagram (Mermaid):
flowchart TD
  A[byte] --> B[short] --> C[int] --> D[long]
  C --> E[float] --> F[double]
Worked Example: the complete Invoice.java below multiplies 199.99 by 3, adds 18% tax, then rounds the final payable amount to two decimal places. Run javac Invoice.java then java Invoice. Expected: Total: 707.96.
Common Mistakes: using double for currency; implicit cast assumptions.
Recap:
• type decisions affect correctness
• numeric operations follow strict conversion rules
• currency needs decimal-safe representation
Practice:
• change quantity to 2 in the supplied invoice. Expected rounded total: 471.98.`,
                code: `import java.math.BigDecimal;

public class Invoice {
    public static void main(String[] args) {
        BigDecimal price = new BigDecimal("199.99");
        BigDecimal qty = new BigDecimal("3");
        BigDecimal taxRate = new BigDecimal("0.18");
        BigDecimal subtotal = price.multiply(qty);
        BigDecimal total = subtotal.add(subtotal.multiply(taxRate))
            .setScale(2, java.math.RoundingMode.HALF_UP);
        System.out.println("Total: " + total);
    }
}`
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
• run the complete Score.java below with 90, -1, and hello on separate runs. Expect Grade: A, Invalid marks, and Enter a whole number. Extension: collect five valid marks before computing an average.`,
                code: `import java.util.Scanner;

public class Score {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter marks (0-100): ");
        if (!sc.hasNextLine()) {
            System.out.println("No input");
            return;
        }
        try {
            int marks = Integer.parseInt(sc.nextLine().trim());
            if (marks < 0 || marks > 100) {
                System.out.println("Invalid marks");
                return;
            }
            String grade = marks >= 90 ? "A" : marks >= 75 ? "B" : marks >= 60 ? "C" : "D";
            System.out.println("Grade: " + grade);
        } catch (NumberFormatException ex) {
            System.out.println("Enter a whole number");
        }
    }
}`
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
Worked Example: start with int[] marks = {60, 90}; int total = 0; for (int mark : marks) total += mark; System.out.println(total / (double) marks.length); inside main. It prints 75.0.
Common Mistakes: god methods; hidden side effects through mutable shared state.
Recap:
• one method = one primary responsibility
• return useful values instead of printing from deep logic
• keep method names verb-focused
Practice:
• move the calculation above into static double average(int[] marks). Reject an empty array with IllegalArgumentException before division, return the average, and print only in main. Verify {60, 90} returns 75.0 and {100} returns 100.0. Do not introduce shared fields.`
            },
            {
                title: "Lesson 5: Arrays, Strings, Enums, and Wrapper Utilities",
                content: `Why this matters: these are the building blocks for most early and interview coding tasks.
Learning Objective: use core APIs safely and efficiently.
Core Theory: arrays have a fixed length and zero-based indices; Java does not specify a physical storage layout for all arrays. String is immutable; enums encode finite domain values; wrappers enable parsing and object use, but unboxing null throws NullPointerException.
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
                code: `                public class StatusDemo {
                    enum Status { NEW, PROCESSING, DONE }
                    public static void main(String[] args) {
                        String[] raw = {"NEW", "DONE", "NEW"};
                        int done = 0;
                        for (String value : raw) {
                            Status s = Status.valueOf(value);
                            if (s == Status.DONE) done++;
                        }
                        System.out.println("Done count: " + done);
                    }
                }`
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
• inside HelloWorld.main run System.out.println(Integer.parseInt("ten"));. Read NumberFormatException and its source line, then change "ten" to "10". Next try new int[]{7}[1], identify ArrayIndexOutOfBoundsException, and use index 0. Expected fixed outputs: 10 and 7.`
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
        this.memberId = java.util.Objects.requireNonNull(memberId, "memberId");
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
• define static void copyNumbers(List<? extends Number> source, List<? super Number> destination) using a for loop that adds each Number to destination. For source List.of(1, 2.5) and an empty List<Object>, expect [1, 2.5].
Run the sum example as Main.java. Expected output: 3.5.`,
                code: `public class Main {
    static double sumNumbers(java.util.List<? extends Number> values) {
        double sum = 0.0;
        for (Number n : values) sum += n.doubleValue();
        return sum;
    }
    public static void main(String[] args) {
        System.out.println(sumNumbers(java.util.List.of(1, 2.5)));
    }
}`
            },
            {
                title: "Lesson 3: Exception Hierarchy and Error Strategy",
                content: `Why this matters: random exception handling makes systems hard to debug and support.
Learning Objective: classify failures and propagate context.
Core Theory: checked exceptions must be caught or declared; unchecked exceptions (RuntimeException subclasses) do not have that compiler requirement. Recovery depends on the operation and caller, not solely on the exception category. Preserve the original cause when wrapping low-level failures.
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
Worked Example: department-wise salary aggregates with collectors. Use the complete Employee record and input fixtures in Guided practice; insert the fragment below inside main after defining employees. Expected averages there: Engineering=150.0 and Sales=90.0. Filter negative salaries as the guide requires before collecting.
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
Core Theory: records have final component fields but are only shallowly immutable: a component can reference a mutable list. Use defensive copies for collection components. Sealed classes constrain inheritance; pattern matching reduces instanceof noise. The switch example requires Java 21 without preview flags.
Diagram (Mermaid):
flowchart TD
  A[sealed PaymentResult] --> B[Success]
  A --> C[Failure]
  A --> D[Pending]
Worked Example: exhaustive switch over payment outcomes. Save the complete program as Main.java; compile with JDK 21+. Expected output: Paid: T1, then Failed: declined on separate lines.
Common Mistakes: using records for mutable entities.
Practice:
• convert DTO class hierarchy to sealed + record structure`,
                code: `public class Main {
    sealed interface PaymentResult permits Success, Failure {}
    record Success(String txnId) implements PaymentResult {}
    record Failure(String reason) implements PaymentResult {}
    static String message(PaymentResult result) {
        return switch (result) {
            case Success s -> "Paid: " + s.txnId();
            case Failure f -> "Failed: " + f.reason();
        };
    }
    public static void main(String[] args) {
        System.out.println(message(new Success("T1")));
        System.out.println(message(new Failure("declined")));
    }
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
Core Theory: a fixed worker count limits active threads, not queued work. Executors.newFixedThreadPool uses an unbounded queue; use ThreadPoolExecutor with a bounded queue and explicit rejection policy for load control. Future/CompletableFuture model results and failures. A timeout alone does not guarantee that underlying work stops; design cooperative cancellation.
Diagram (Mermaid):
flowchart LR
  A[tasks] --> B[bounded queue]
  B --> C[fixed pool workers]
  C --> D[future results]
Practice:
• run this complete Main.java example; expect AB. Then make the second task throw IllegalStateException("B failed") and observe join propagating a CompletionException while finally still requests shutdown. Use the guided exercise for per-file counts.`,
                code: `import java.util.concurrent.*;
public class Main {
    public static void main(String[] args) {
        ExecutorService pool = new ThreadPoolExecutor(2, 2, 0L, TimeUnit.MILLISECONDS,
            new ArrayBlockingQueue<>(2), new ThreadPoolExecutor.AbortPolicy());
        try {
            CompletableFuture<String> first = CompletableFuture.supplyAsync(() -> "A", pool);
            CompletableFuture<String> second = CompletableFuture.supplyAsync(() -> "B", pool);
            System.out.println(first.thenCombine(second, (a, b) -> a + b).join());
        } finally { pool.shutdown(); }
    }
}`
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
Core Theory: java.nio.file APIs support paths and buffered streaming. For integrity-sensitive writes, write a sibling staging file, close it, then request an atomic move on the same filesystem. ATOMIC_MOVE may be unsupported; report that explicitly rather than promising atomicity after a non-atomic fallback. Atomic visibility does not itself guarantee durability after power loss.
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
        description: "Run a Python script, then build confidence with values, decisions, loops, functions, collections, and tracebacks.",
        duration: "85 min",
        lessons: "14 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Setup and First Script", "Variables and Dynamic Typing", "Operators and Expressions", "Control Flow", "Functions", "Scope Rules", "Strings", "Lists", "Tuples", "Sets", "Dictionaries", "CLI and Debugging Basics", "Code Style", "Optional: History and CPython Bytecode"],
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
                                title: "Optional: Python Introduction and History",
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
                title: "Lesson 1: Install Python and Run a Script",
                content: `Goal: run a small program before studying how the interpreter works.
1. Install Python 3.11 or later from a trusted source. Open a terminal and run python --version (on Windows, py --version is another option; on some systems use python3).
2. Create a learning folder and save hello.py containing print("Hello, Python!"). Use a plain-text editor, not a word processor.
3. Open a terminal in that folder and run python hello.py. Use your verified interpreter command consistently. Expected output: Hello, Python!
4. Add score = 75 and print(score + 5). Save and rerun; the second line should be 80.
5. Add if score >= 60: followed by an indented print("Pass") on the next line. Use four spaces for indentation. Expected final line: Pass.
6. Add for score in [60, 75, 90]: with an indented print(score). It prints those three numbers on separate lines. input("Score: ") returns text; int("75") converts text to an integer, while int("hello") raises ValueError.
Practice: print your name, then print(7 + 3). Explain why print("7" + "3") instead prints 73.
Optional, after functions and tracebacks: CPython compiles source into bytecode and executes it with its interpreter. Imported modules may cache bytecode in __pycache__; running a script does not necessarily create that cache for the main file. Other Python implementations differ. To explore, define def twice(n): return n * 2, then use import dis; dis.dis(twice). Opcode names vary by Python version; memorizing them is not a prerequisite.`
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
Worked Example: a = [1]; b = a; b.append(2); print(a) prints [1, 2]. Replace b = a with b = a.copy(); now a stays [1]. A shallow copy still shares nested objects.
Common Mistakes: using mutable default arguments.
Recap:
- names reference objects
- copying and mutation are separate operations
- immutability simplifies reasoning
Practice:
- repair def remember(value, seen=[]): seen.append(value); return seen. Call remember("A"), then remember("B"). They currently share a list; use seen=None and create [] inside when seen is None so the calls return ["A"] and ["B"] independently.`
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
Core Theory: LEGB means Local, Enclosing, Global, Built-in name lookup. A parameter is a local name. return sends a result to the caller; print only displays text. Prefer pure functions (same inputs, same result, no external mutation) for calculations.
Diagram (Mermaid):
flowchart TD
    A[input] --> B[validate]
    B --> C[transform]
    C --> D[format result]
Common Mistakes: giant functions and hidden global dependencies.
Practice:
- start with the supplied compute_grade function, then write summarize(scores) that returns [compute_grade(s) for s in scores]. Call print(summarize([59, 60, 90])); expected ['D', 'C', 'A']. Keep input and printing outside compute_grade.`,
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
- replace def discount(price, percent=10) with def discount(price, *, percent=10). Return price * (1 - percent / 100). Verify discount(100, percent=20) returns 80.0 and discount(100, 20) raises TypeError. Validate nonnegative price and percent between 0 and 100 before calculating.`
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
    amount: int
    currency: str

    def __post_init__(self):
        if type(self.amount) is not int or self.amount < 0:
            raise ValueError("amount must be non-negative integer minor units")
        if self.currency not in {"USD", "EUR"}:
            raise ValueError("unsupported currency")

# Amount is in cents: Money(1250, "USD") means 12.50 USD.
print(Money(1250, "USD"))`
            },
            {
                title: "Lesson 2: Composition over Inheritance",
                content: `Why this matters: inheritance misuse creates rigid class hierarchies.
Learning Objective: identify when composition is safer and easier to evolve.
Core Theory: favor has-a relationships for behavior assembly.
Practice:
- create Checkout(policy) where policy is a callable receiving integer cents. Compare Checkout(lambda cents: cents).total(1000) with Checkout(lambda cents: cents * 9 // 10).total(1000): expect 1000 and 900. Checkout.total delegates to its policy; no subclass is needed. Use only nonnegative integer cents for this exercise.`
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
3) dataclass(frozen=True) normally prevents: A) mutation of every nested object B) field reassignment C) constructor creation D) type checking
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
Core Theory: after __enter__ succeeds, with calls __exit__ when the block exits normally or by an exception. Cleanup is not guaranteed if entry fails or the process is forcibly terminated. Returning a truthy value from __exit__ suppresses the exception, so do so only intentionally.
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
                content: `                Why this matters: type hints help static tools catch integration mistakes before runtime; annotations themselves do not enforce types. Validate untrusted input separately.
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
                content: `1) After successful context entry, with normally invokes: A) a compiler B) the context exit handler C) no exceptions D) faster loops
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
Core Theory: on standard GIL-enabled CPython, CPU-bound Python bytecode usually cannot execute in parallel across threads. Processes can run it in parallel, with startup/serialization costs. Native extensions may release the GIL, and optional free-threaded builds differ. asyncio coordinates cooperative I/O; a blocking call inside a coroutine still blocks its event loop.
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
- run the complete fake-client test below as a script. It should report one passing test and exactly two fetch calls. Then compare with the async retry tests in Guided practice.`,
                code: `import unittest
from unittest.mock import Mock

class DataService:
    def __init__(self, client, max_retries=1):
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self.client = client
        self.max_retries = max_retries
    def load(self):
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.fetch()
            except TimeoutError:
                if attempt == self.max_retries:
                    raise

class TestService(unittest.TestCase):
    def test_retries_on_timeout(self):
        client = Mock()
        client.fetch.side_effect = [TimeoutError(), {"ok": True}]
        service = DataService(client, max_retries=1)
        result = service.load()
        self.assertTrue(result["ok"])
        self.assertEqual(client.fetch.call_count, 2)

if __name__ == "__main__":
    unittest.main()`
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
                content: `1) For CPU-heavy pure Python on GIL-enabled CPython, which can use multiple cores? A) threading alone B) multiprocessing C) asyncio alone D) recursion alone
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
        description: "Create and run a console app, then learn C# values, decisions, methods, collections, and error messages.",
        duration: "85 min",
        lessons: "14 lessons",
        isNew: true,
        isLocked: false,
        topics: ["SDK Setup and First Console App", "Variables and Operators", "Control Flow", "Methods", "Value vs Reference Types", "Parameter Modifiers", "Strings", "Arrays", "Collections Basics", "Namespaces", "Project and Build Basics", "Exception Basics", "Debugging with Stack Traces", "Optional: History, CLR and IL"],
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
                                title: "Optional: C# Introduction and History",
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
                title: "Lesson 1: Install the SDK and Run a Console App",
                content: `Goal: get visible output from your own program.
1. Install a supported .NET SDK (not just the runtime). In a new terminal run dotnet --list-sdks. A version line confirms an SDK is available.
2. In a learning folder run dotnet new console -n HelloCourse, then cd HelloCourse. The project file records your target framework; Program.cs holds your source.
3. Replace Program.cs entirely with the complete example below. Do not append it to the template's top-level statements.
4. Run dotnet run. Expected output: Hello from C# runtime. This command builds before it runs.
5. Inside Main add int score = 75; Console.WriteLine(score + 5);. The next line is 80. Add if (score >= 60) { Console.WriteLine("Pass"); } to print Pass.
6. Add foreach (int mark in new[] {60, 75, 90}) { Console.WriteLine(mark); }. Predict the three printed numbers before running.
Practice: print your name and the result of 7 + 3 on separate lines. A string uses double quotes; an int is a whole number; braces group code; semicolons end statements.
If dotnet is missing, reopen the terminal after SDK installation. If a build fails, fix the first diagnostic and rerun.
Optional runtime context after these basics: normal SDK builds emit IL and metadata in assemblies. The CLR manages execution and the JIT commonly compiles methods to native code on demand. Native AOT is an alternative deployment mode, so C# is not restricted to JIT execution.`,
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
- reference assignment copies a reference to the same object; value/reference behavior is not a simple stack/heap distinction
- nullability annotations improve API clarity`
            },
            {
                title: "Lesson 3: Methods, Parameter Passing, and Scope",
                content: `Why this matters: method contract quality directly affects maintainability.
Learning Objective: design clear signatures using value, ref, out, and optional parameters responsibly.
Core Theory: default passing is by value; ref/out modify caller-visible variables and should be used intentionally.
Practice:
- use the supplied TryParseScore in Program. Call it with "75", "-1", and "hello". Expect true/75, false/0, false/0. Print only after checking its boolean result; zero on failure is not a valid parsed score.`,
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
- put int[] values = Array.Empty<int>(); Console.WriteLine(values.First()); in Program.cs (add using System.Linq;). First() on an empty sequence throws InvalidOperationException. Check values.Length first and print "No scores" instead. Verify an array containing 42 prints 42.`
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
                content: `1) A normal .NET SDK build emits: A) native code only, always B) IL and metadata C) JSON only D) XML source
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
- create a synchronous IPricePolicy with decimal Apply(decimal subtotal). Implement FullPrice (return subtotal) and TenPercentOff (return subtotal * 0.9m). Inject the policy into Checkout. For 100m expect 100m and 90m respectively; reject negative subtotals before invoking either policy. Async collaborators are deferred to Module 4.`
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
Core Theory: an async method runs synchronously until it reaches an incomplete await. It then returns control to its caller; await does not itself create a thread. Completed tasks may continue synchronously. Continuation context depends on the host; console apps normally have no synchronization context.
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
- insert the FetchAsync method below into a class with System.Net.Http, System.Threading, and System.Threading.Tasks imports. It requires a caller-owned HttpClient and a real URL; it is an integration fragment, not a complete console program. For a dependency-free, runnable introduction use the two-task cancellation example in Guided practice instead. Never use a live network response as a deterministic unit test.`,
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
        description: "Compile one small C++ program, then learn initialization, decisions, functions, input, and diagnostics before multi-file builds.",
        duration: "90 min",
        lessons: "14 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Compiler Setup and First Program", "Types and Initialization", "Control Flow", "Input/Output", "Functions", "Const Correctness", "References", "Namespaces", "Header and Source Separation", "Compilation and Linking", "Error Messages", "Build Configurations", "Assertions", "Debugging Basics", "Optional: C++ History"],
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
                                title: "Optional: C++ Introduction and History",
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
                title: "Lesson 1: Set Up a Compiler and Run C++",
                content: `Goal: build a single-file program before splitting code across files.
1. Choose one toolchain. On Windows, install Visual Studio Build Tools with Desktop development with C++ and open its Developer PowerShell. Run cl to check it is available. Alternatively use an existing GCC/Clang toolchain and check g++ --version or clang++ --version.
2. Create a learning folder. Save the complete example as main.cpp, not main.cpp.txt.
3. For MSVC run cl /std:c++20 /EHsc /W4 main.cpp /Fe:hello.exe. For GCC run g++ -std=c++20 -Wall -Wextra -pedantic main.cpp -o hello.exe (Clang accepts equivalent flags).
4. In Windows PowerShell run .\\hello.exe. Expected output: Hello from C++ toolchain.
5. Inside main add int score{75}; std::cout << score + 5 << std::endl;. Rebuild and run; the next line is 80. Add if (score >= 60) { std::cout << "Pass" << std::endl; }.
6. A loop for (int i{1}; i <= 3; ++i) { std::cout << i << std::endl; } prints 1, 2, 3. std::cin >> score reads a whole number; always test whether that extraction succeeded.
Read the code: #include makes a standard-library declaration available; main is the entry point; std::cout writes output; return 0 signals success. Save and rebuild after each edit.
Practice: print your name, then 7 + 3. Expected second line: 10.
Later build model: preprocessing expands includes/macros; compilation checks each translation unit and emits object code; linking resolves definitions across object files. Missing definitions are linker errors, not input-validation failures.`,
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
Core Theory: brace initialization rejects narrowing conversions such as int count{2.5}; initialize variables before reading them. const prevents modification through that name; const references do not make a separately aliased object deeply immutable.
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
                code: `#include <iostream>
#include <vector>

int sum(const std::vector<int>& values) {
    int total = 0;
    for (int v : values) total += v;
    return total;
}

int main() {
    std::cout << sum({60, 90}) << std::endl; // 150; inputs here fit in int.
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
Core Theory: raw pointers may be nullable and rebindable. A valid reference aliases an existing object and cannot be reseated; it does not extend arbitrary object lifetimes or prevent dangling. Raw pointers and references are normally borrowers, not owners.
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
Core Theory: RAII ties release to an owning object's destructor during normal scope exit and exception unwinding. Destructors should not throw. Forced process termination and undefined behavior are outside this guarantee. Standard streams, containers, and smart pointers already provide RAII.
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
                code: `#include <algorithm>
#include <iostream>
#include <vector>

int main() {
    std::vector<int> values{5, 2, 8, 1};
    std::sort(values.begin(), values.end());
    std::vector<int> doubled(values.size());
    std::transform(values.begin(), values.end(), doubled.begin(),
                   [](int v) { return v * 2; });
    for (int value : doubled) std::cout << value << ' ';
    std::cout << std::endl; // 2 4 10 16 (with a trailing space).
}`
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
Core Theory: std::move is a cast that permits move overload selection; it does not itself transfer anything. A move constructor/assignment may transfer resources; a const source may instead be copied. Moved-from standard-library objects are valid but often have unspecified values. Do not assume a moved-from vector is empty; a moved-from unique_ptr is null.
Practice:
- profile copy-heavy code path and reduce copies with move-aware design`,
                code: `#include <iostream>
#include <vector>

std::vector<int> buildData() {
    std::vector<int> v(1000, 1);
    return v; // NRVO/move
}
int main() {
    auto data = buildData();
    std::cout << data.size() << ' ' << data.front() << std::endl; // 1000 1
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
Core Theory: conflicting unsynchronized accesses to the same memory, with at least one write, form a data race and cause undefined behavior in C++. This is stronger than merely obtaining an occasionally wrong count. Use mutexes or atomics and join threads before consuming their results.
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
                code: `#include <mutex>

std::mutex m1, m2;

void safeWork() {
    std::scoped_lock lock(m1, m2);
    // Every caller accessing the protected pair must hold these locks.
}
int main() { safeWork(); }`
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

// Guides are separate from lessons. Exact titles make accidental curriculum drift visible.
(() => {
    const guides = {
        javaProgramming: {
            "Java Core and Toolchain": {
                prerequisites: "None. A terminal runs commands; a source file is plain text containing instructions. Start with the setup and first-program lesson, not bytecode inspection.",
                setup: "Install JDK 21 or later. Check java -version and javac -version in a new terminal. In an empty learning folder save the solution as Main.java. Run javac Main.java, then java Main 75. Arguments follow Main; do not type a .class suffix. Recompile after edits.",
                workedExample: {
                    problem: "Turn one whole-number score into a grade: A for 90–100, B for 75–89, C for 60–74, D for 0–59.",
                    steps: [
                        "Start with public class Main { public static void main(String[] args) { System.out.println(\"Hello\"); } }. Save as Main.java, compile, and run java Main. Confirm Hello before adding input.",
                        "Replace the greeting with int score = 75; System.out.println(score >= 60 ? \"Pass\" : \"Fail\");. This prints Pass: >= includes the boundary.",
                        "In the solution, args[0] is text. Check that exactly one argument exists, parse it inside try, reject scores outside 0–100, then test grade thresholds from highest to lowest. For 75 the >=90 check fails and >=75 succeeds."
                    ],
                    expected: "java Main 75 prints B. java Main 60 prints C. Invalid or missing input prints Invalid score rather than a stack trace."
                },
                practice: {
                    task: "Implement that exact single-score command-line grader, accepting an integer from 0 through 100 inclusive. Print only the grade or Invalid score; handle missing, extra, non-integer, and out-of-range arguments.",
                    cases: [
                        {input: "java Main 90", expected: "A"},
                        {input: "java Main 59", expected: "D"},
                        {input: "java Main hello", expected: "Invalid score"},
                        {input: "java Main 101", expected: "Invalid score"}
                    ],
                    hint: "Check args.length before indexing. Integer.parseInt throws NumberFormatException for text that is not an int. A series of descending thresholds avoids overlapping branches.",
                    solution: `public class Main {
    public static void main(String[] args) {
        if (args.length != 1) { System.out.println("Invalid score"); return; }
        try {
            int score = Integer.parseInt(args[0]);
            if (score < 0 || score > 100) { System.out.println("Invalid score"); return; }
            String grade = score >= 90 ? "A" : score >= 75 ? "B" : score >= 60 ? "C" : "D";
            System.out.println(grade);
        } catch (NumberFormatException ex) {
            System.out.println("Invalid score");
        }
    }
}
// The length guard prevents an invalid index; parsing and range validation are separate checks. Only valid scores reach classification.`
                },
                interview: {question: "What is the difference between compiling and running this program?", answer: "javac checks Main.java and produces Main.class bytecode. java Main starts the JVM and invokes main. Editing source requires recompilation; running an old class file can otherwise hide your change."},
                readiness: ["I can create, compile, change, and rerun Main.java without an IDE.", "I can trace 59, 60, 74, 75, 89, and 90 through the branches and explain both invalid-input paths."]
            },
            "Object-Oriented Design and Architecture": {
                prerequisites: "Complete Java Module 1: methods, private/local state, conditionals, and exceptions. A constructor creates an object's initial valid state; an invariant is a rule that must keep holding.",
                setup: "Use JDK 21+. Save the complete solution as Main.java; run javac Main.java then java Main. No libraries or build tool are needed. Amounts below are integer cents, not floating-point currency.",
                workedExample: {
                    problem: "Protect a wallet from spending more cents than it owns.",
                    steps: [
                        "Construct Wallet(500). Its private cents field starts at 500. Reject a negative opening balance in the constructor so callers cannot create an invalid wallet.",
                        "Call spend(200). Check amount > 0 and amount <= cents before subtracting. The new balance is 300.",
                        "Call spend(400). Return false without mutation. A balance() query must still return 300; the caller never writes the field directly."
                    ],
                    expected: "For the worked calls: true, false, 300 on separate lines. A rejected operation leaves the invariant cents >= 0 intact."
                },
                practice: {
                    task: "Implement Wallet with a validated constructor, boolean spend(int amount), and int balance(). Reject non-positive spends and overspending without changing state. Demonstrate two independent wallets.",
                    cases: [
                        {input: "Wallet(500); spend(200); spend(400); balance()", expected: "true, false, 300"},
                        {input: "Wallet(100); spend(0); spend(100); balance()", expected: "false, true, 0"},
                        {input: "Wallet(-1)", expected: "IllegalArgumentException"}
                    ],
                    hint: "Make cents private. Test all rejection conditions before the single mutation. Do not add a public setter that bypasses the rule.",
                    solution: `public class Main {
    static final class Wallet {
        private int cents;
        Wallet(int cents) {
            if (cents < 0) throw new IllegalArgumentException("negative balance");
            this.cents = cents;
        }
        boolean spend(int amount) {
            if (amount <= 0 || amount > cents) return false;
            cents -= amount;
            return true;
        }
        int balance() { return cents; }
    }
    public static void main(String[] args) {
        Wallet a = new Wallet(500);
        System.out.println(a.spend(200));
        System.out.println(a.spend(400));
        System.out.println(a.balance());
        Wallet b = new Wallet(100);
        System.out.println(b.spend(0));
        System.out.println(b.spend(100));
        System.out.println(b.balance());
    }
}
// Each instance owns separate state. The constructor validates creation; spend validates each transition, so no caller can bypass the balance rule.`
                },
                interview: {question: "Why not make cents public and check it in main?", answer: "Another caller could bypass main's checks. Encapsulation puts the invariant beside the state in every construction/update path. Interfaces become useful when a service genuinely needs interchangeable wallet/storage behavior, not merely to increase class count."},
                readiness: ["I can show that a rejected spend does not change either wallet.", "I can explain which rules belong in a constructor versus an operation, and why composition does not require inheritance."]
            },
            "Data Structures, Generics, and Reliability": {
                prerequisites: "Java Modules 1–2, loops, classes, exceptions, and the difference between a list and an ID lookup. Map<K,V> associates a key with one value.",
                setup: "JDK 21+, Main.java, javac Main.java then java Main. The supplied importer uses an in-memory String array, so no CSV parser or files are needed. Its deliberately simple format is id,integerScore with no quoted commas.",
                workedExample: {
                    problem: "Import S1,80 then S1,90 without silently replacing the first student.",
                    steps: [
                        "Split each row with split(\",\", -1), trim both fields, require a nonempty ID and an integer score between 0 and 100. Validate before touching the map.",
                        "Insert S1 -> 80. When the second row's ID is already present, record a duplicate reject instead of calling put again.",
                        "Continue after the reject. Use TreeMap for an ID-sorted final display; collect rejection messages separately with one-based row numbers."
                    ],
                    expected: "Stored record: S1=80. Reject: row 2: duplicate. Counts are accepted=1, rejected=1."
                },
                practice: {
                    task: "Import the cases below. Keep the first valid occurrence of each ID, reject malformed rows/ranges/duplicates with a reason and row number, and print sorted stored records plus reject count. An invalid occurrence must not reserve its ID.",
                    cases: [
                        {input: "[S1,80; S1,90; S2,101; S3,70]", expected: "{S1=80, S3=70}; rejects=2; row 2 duplicate, row 3 range"},
                        {input: "[S1,x; S1,60; ,50; bad]", expected: "{S1=60}; rejects=3; rows 1 number, 3 id, 4 fields"}
                    ],
                    hint: "Use one rejection per row. Check fields, ID, number, range, then duplicates in that order. Catch only numeric conversion failures.",
                    solution: `import java.util.*;
public class Main {
    static void load(String[] rows) {
        Map<String, Integer> scores = new TreeMap<>();
        List<String> rejects = new ArrayList<>();
        for (int i = 0; i < rows.length; i++) {
            String[] p = rows[i].split(",", -1);
            String reason = null;
            if (p.length != 2) reason = "fields";
            else if (p[0].trim().isEmpty()) reason = "id";
            else {
                try {
                    int score = Integer.parseInt(p[1].trim());
                    String id = p[0].trim();
                    if (score < 0 || score > 100) reason = "range";
                    else if (scores.containsKey(id)) reason = "duplicate";
                    else scores.put(id, score);
                } catch (NumberFormatException ex) { reason = "number"; }
            }
            if (reason != null) rejects.add("row " + (i + 1) + ": " + reason);
        }
        System.out.println(scores);
        System.out.println("rejects=" + rejects.size());
        rejects.forEach(System.out::println);
    }
    public static void main(String[] args) {
        load(new String[]{"S1,80", "S1,90", "S2,101", "S3,70"});
        load(new String[]{"S1,x", "S1,60", ",50", "bad"});
    }
}
// Validation completes before insertion, so a bad row cannot overwrite data or reserve an ID. TreeMap gives O(log n) operations and sorted output.`
                },
                interview: {question: "When would HashMap be preferable, and what does PECS mean?", answer: "Use HashMap for expected O(1) lookup when sorted traversal is not required; explicitly sort only at reporting time. Producer Extends, Consumer Super describes wildcard APIs: read Number from List<? extends Number>, and write Integer to List<? super Integer>. It is unrelated to exception recovery."},
                readiness: ["I can trace each rejected row and prove accepted + rejected equals input row count.", "I can explain sorted versus hash lookup costs and keep invalid rows from mutating indexes."]
            },
            "Modern Java: Streams to Pattern Matching": {
                prerequisites: "Java Modules 1–3: typed collections, lambdas as behavior passed to a method, and basic loops. Records and sealed switch examples in this module target Java 21.",
                setup: "Use JDK 21+ with javac Main.java and java Main. The solution is a complete file. Salary units are integer cents; averages are displayed as doubles for this exercise, not used to settle payments.",
                workedExample: {
                    problem: "Average salaries by department without mutating an external total from inside a stream.",
                    steps: [
                        "Represent each row with record Employee(String dept, int salary). Prepare Engineering salaries 100 and 200 and Sales salary 90.",
                        "Filter out negative salaries, then collect with groupingBy(Employee::dept, TreeMap::new, averagingInt(Employee::salary)). Each department gets its own accumulator.",
                        "The Engineering sum is 300 across two accepted rows, giving 150.0. TreeMap keeps output sorted; the input list is unchanged."
                    ],
                    expected: "{Engineering=150.0, Sales=90.0}. A department with only rejected rows is absent, not assigned a fabricated zero average."
                },
                practice: {
                    task: "Write averages(List<Employee>) with the exact filtering/grouping behavior above. Empty input must yield an empty map. Keep the original list unchanged and include tests for a negative salary.",
                    cases: [
                        {input: "Engineering:100, Engineering:200, Sales:90, Sales:-1", expected: "{Engineering=150.0, Sales=90.0}"},
                        {input: "empty list", expected: "{}"},
                        {input: "Sales:-1", expected: "{}"}
                    ],
                    hint: "Use a terminal collect operation; merely building a stream does not run its pipeline. Supply TreeMap::new if output order matters.",
                    solution: `import java.util.*;
import java.util.stream.Collectors;
public class Main {
    record Employee(String dept, int salary) {}
    static Map<String, Double> averages(List<Employee> employees) {
        return employees.stream().filter(e -> e.salary() >= 0)
            .collect(Collectors.groupingBy(Employee::dept, TreeMap::new,
                Collectors.averagingInt(Employee::salary)));
    }
    public static void main(String[] args) {
        System.out.println(averages(List.of(new Employee("Engineering", 100),
            new Employee("Engineering", 200), new Employee("Sales", 90),
            new Employee("Sales", -1))));
        System.out.println(averages(List.of()));
        System.out.println(averages(List.of(new Employee("Sales", -1))));
    }
}
// The collector owns aggregation state. List.of also makes accidental list mutation fail; records with mutable components would still need defensive copies.`
                },
                interview: {question: "Why not accumulate salaries into a shared map inside forEach?", answer: "External mutation hides dependencies and can become unsafe if parallel execution is introduced. A collector expresses grouping and accumulation explicitly. Streams are not automatically faster than loops; compare readability and measure relevant workloads."},
                readiness: ["I can predict when the pipeline executes and explain why negative-only groups disappear.", "I can write an equivalent loop and verify identical empty/grouped outputs before replacing legacy code."]
            },
            "Concurrency, IO, and JVM Internals": {
                prerequisites: "Java Modules 1–4, exceptions and try/finally. Understand that two tasks may overlap and that joining a result is different from creating a task.",
                setup: "JDK 21+, Main.java, javac Main.java then java Main. Start with in-memory rows to isolate concurrency from filesystem faults. The two-worker executor below is intentionally bounded; no network access is required.",
                workedExample: {
                    problem: "Count valid score lines concurrently while preserving the same summary as a serial scan.",
                    steps: [
                        "Treat each file fixture as a separate List<String>: [80, bad, 60] yields accepted=2, rejected=1; [101, 90] yields accepted=1, rejected=1.",
                        "Let each worker return its own immutable Counts record. Submit at most these two tasks, wait for both futures, then add their counts on the caller thread.",
                        "Use a bounded queue and shut down in finally. If a future fails, report the operation failure rather than pretending that a zero-count file succeeded."
                    ],
                    expected: "accepted=3 rejected=2 on every completed run. Worker completion order must not affect this line."
                },
                practice: {
                    task: "Implement the two-fixture processor below with no shared mutable counters. Count integers 0–100 as accepted and all other lines as rejected. An empty fixture contributes zero. Preserve cleanup if a future throws.",
                    cases: [
                        {input: "[80, bad, 60] and [101, 90]", expected: "accepted=3 rejected=2"},
                        {input: "[] and [-1, x]", expected: "accepted=0 rejected=2"}
                    ],
                    hint: "Return per-task results, then sum after get(). Bounding active workers alone is not queue backpressure. A production reader must additionally close file streams and support cooperative cancellation.",
                    solution: `import java.util.*;
import java.util.concurrent.*;
public class Main {
    record Counts(int accepted, int rejected) {}
    static Counts scan(List<String> lines) {
        int good = 0;
        for (String line : lines) {
            try {
                int n = Integer.parseInt(line.trim());
                if (n >= 0 && n <= 100) good++;
            } catch (NumberFormatException ex) { /* rejected below */ }
        }
        return new Counts(good, lines.size() - good);
    }
    static void run(List<String> a, List<String> b) throws Exception {
        ExecutorService pool = new ThreadPoolExecutor(2, 2, 0L, TimeUnit.MILLISECONDS,
            new ArrayBlockingQueue<>(2), new ThreadPoolExecutor.AbortPolicy());
        try {
            Future<Counts> x = pool.submit(() -> scan(a));
            Future<Counts> y = pool.submit(() -> scan(b));
            Counts p = x.get(), q = y.get();
            System.out.println("accepted=" + (p.accepted() + q.accepted())
                + " rejected=" + (p.rejected() + q.rejected()));
        } finally { pool.shutdownNow(); }
    }
    public static void main(String[] args) throws Exception {
        run(List.of("80", "bad", "60"), List.of("101", "90"));
        run(List.of(), List.of("-1", "x"));
    }
}
// Only the caller combines results. No worker writes another worker's counts, so there is no lost-update race. shutdownNow requests interruption; real long-running tasks must honor it.`
                },
                interview: {question: "Is ConcurrentHashMap enough to make a multi-step operation atomic?", answer: "No. Individual operations are thread-safe, but a read-then-write sequence can still race. Use atomic map methods such as compute for a per-key update or a lock/transaction for invariants spanning keys. Measure contention before choosing a more complex design."},
                readiness: ["I can compare serial and concurrent totals over repeated runs and explain executor cleanup.", "I can distinguish a bounded worker count, a bounded queue, a timeout, and cooperative cancellation."]
            },
            "Production Java Capstone and Testing": {
                prerequisites: "Java Modules 1–5. Be able to validate domain operations, isolate repositories, run tests, and explain a transaction. This is a multi-session capstone, not a one-file exercise.",
                setup: "Choose the Banking Ledger option with JDK 21+, Maven or Gradle, JUnit 5, and a local transactional SQL database/JDBC driver. Record exact versions and a repeatable test command in the project README. First implement an in-memory repository and deterministic tests before connecting JDBC; no cloud account is needed.",
                workedExample: {
                    problem: "Transfer 250 cents from account A (1000) to B (100) without creating or losing money.",
                    steps: [
                        "Define transfer(requestId, fromId, toId, cents). Require distinct existing accounts, cents > 0, and sufficient funds. Initial total is 1100 cents.",
                        "In one database transaction, claim a unique requestId, conditionally debit A if its balance is sufficient, and credit B. If any update fails, roll back both updates and the claim.",
                        "Commit the successful result with requestId T1. Repeat T1 with identical payload: return the stored result, not another debit. Reject reuse of T1 with a different payload."
                    ],
                    expected: "After T1: A=750, B=350, total=1100. Replaying T1 keeps these balances. An injected credit failure leaves A=1000, B=100 on a fresh fixture."
                },
                practice: {
                    task: "Deliver the concrete ledger workflow above behind app/service/domain/repository boundaries. Acceptance: integer cents; no negative balances; durable idempotency; rollback on a forced credit failure; persisted balances after restart; >=15 automated tests including concurrency, duplicates, invalid input and JDBC integration. Include setup/run/test commands, an ADR on transaction strategy, and a five-minute scripted demo.",
                    cases: [
                        {input: "Fresh A=1000 B=100; T1 transfer A->B 250; replay T1", expected: "A=750 B=350; one committed transfer"},
                        {input: "Fresh A=1000 B=100; T2 transfer 1200", expected: "insufficient funds; no balance or ledger mutation"},
                        {input: "Fresh fixture; force credit update to fail after debit", expected: "transaction rolled back; A=1000 B=100; no committed request"}
                    ],
                    hint: "A service-level check without a database transaction does not protect concurrent transfers. A unique request ID must be persisted in the same atomic unit as the balances.",
                    solution: "Implementation route: define Money as validated integer cents and TransferResult as an immutable value. Give the repository a transfer transaction operation rather than separate public debit/credit calls. Inside JDBC set autoCommit(false); insert a unique request row carrying the payload; debit using UPDATE ... WHERE balance >= amount; require one affected row; credit using an overflow-safe update; store result; commit. On any failure roll back and preserve the cause. Handle duplicate request IDs by loading the existing payload/result in a clean transaction and comparing all fields. Use consistent row-lock order or a documented retry strategy for deadlocks. Test with two connections attempting incompatible debits and assert no negative balance and unchanged total. Parameterize every SQL value, use try-with-resources, and verify restart persistence against the database, not the in-memory fake. A mock alone cannot prove rollback or isolation."
                },
                interview: {question: "What evidence would justify claiming this ledger is reliable?", answer: "A reproducible build, domain tests, real-database rollback/isolation tests, duplicate-request tests, restart verification, and documented failure limits. Passing fifteen tests alone is not proof of production correctness; explain the database isolation assumptions and unresolved operational risks."},
                readiness: ["I can run the accepted, rejected, replay, and forced-failure scenarios from a fresh database and show balances.", "I can present test results and the transaction ADR, including what was not load-tested or operationally validated."]
            }
        },
        pythonProgramming: {
            "Python Core and Runtime Foundations": {
                prerequisites: "None. You need a text editor and a terminal. A variable names a value; a function is a reusable group of instructions.",
                setup: "Install Python 3.11+ and verify python --version (or py --version on Windows). Save the solution as grades.py in a learning folder; run python grades.py using the command you verified. No packages are required. Indent function bodies with four spaces.",
                workedExample: {
                    problem: "Classify a score and distinguish numeric input from invalid text.",
                    steps: [
                        "Run print(\"Hello, Python!\") first and check the message. Next score = 75; print(score + 5) prints 80, but input() returns text, not an integer.",
                        "Define grade(text). Try int(text); if conversion raises ValueError, return Invalid score. Then reject values below 0 or above 100.",
                        "For text \"75\", int yields 75. The >=90 branch fails, the >=75 branch succeeds, and return \"B\" ends the function. print(grade(\"75\")) displays that returned value."
                    ],
                    expected: "Calling grade with \"75\", \"59\", and \"hello\" returns B, D, and Invalid score respectively; each print displays one line."
                },
                practice: {
                    task: "Write grade(text) using A=90–100, B=75–89, C=60–74, D=0–59. Accept only text convertible by int(), then validate range. Return Invalid score for invalid text or range. Use the supplied list to run all examples.",
                    cases: [
                        {input: "grade('90')", expected: "A"},
                        {input: "grade('60')", expected: "C"},
                        {input: "grade('75.5')", expected: "Invalid score"},
                        {input: "grade('-1')", expected: "Invalid score"}
                    ],
                    hint: "Only put conversion in try/except ValueError; do not catch every exception. Return results from the function and print at the call site.",
                    solution: `def grade(text):
    try:
        score = int(text)
    except ValueError:
        return "Invalid score"
    if not 0 <= score <= 100:
        return "Invalid score"
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    return "D"

for text in ["90", "60", "75.5", "-1"]:
    print(grade(text))

# The function returns on the first matching branch. Converting text and checking its numeric range are different operations; neither silently clamps an invalid score.`
                },
                interview: {question: "Does b = a copy a list?", answer: "No. Both names refer to the same list, so appending through b changes what a sees. a.copy() creates a shallow outer-list copy; nested mutable objects are still shared."},
                readiness: ["I can save and rerun a script, identify indentation errors, and distinguish print from return.", "I can explain why '75' needs conversion, why 75.5 text is rejected, and what happens at each grade boundary."]
            },
            "Functions, Comprehensions, and Data Pipelines": {
                prerequisites: "Python Module 1: functions, loops, lists, dictionaries, int conversion, and targeted exception handling. A generator yields one result at a time instead of constructing a result list.",
                setup: "Python 3.11+, save solution as pipeline.py and run python pipeline.py. The example uses an iterator of strings rather than a real file. Later wrap a CSV reader in with open(..., newline='', encoding='utf-8') for quoted CSV handling.",
                workedExample: {
                    problem: "Read score text once, yield valid scores, and preserve exact rejection reasons.",
                    steps: [
                        "For enumerate(rows, start=1), convert each stripped string with int(). Rows \"90\", \"oops\", and \"60\" convert to 90, conversion failure, and 60.",
                        "Append (2, 'number') to rejects for oops; yield only 90 and 60. Do not build a full list of all raw rows.",
                        "Consume the generator with sum(score >= 60 for score in valid_scores(...)). Both yielded scores pass, so the total is 2. Side effects in rejects occur during consumption, not generator creation."
                    ],
                    expected: "passes=2 rejects=[(2, 'number')]. Empty input produces passes=0 rejects=[]."
                },
                practice: {
                    task: "Implement valid_scores(rows, rejects) as a lazy generator over a one-pass input iterator. Accept 0–100 inclusive. Store (one-based row number, 'number' or 'range') for every reject, then count scores >=60 without consuming the iterator twice.",
                    cases: [
                        {input: "['90', 'oops', '60', '101']", expected: "passes=2 rejects=[(2, 'number'), (4, 'range')]"},
                        {input: "['0', '59', '100']", expected: "passes=1 rejects=[]"},
                        {input: "[]", expected: "passes=0 rejects=[]"}
                    ],
                    hint: "yield inside a function makes a generator. Use continue after conversion failure so rejected text never reaches the range check. A reject list still consumes memory proportional to rejects.",
                    solution: `def valid_scores(rows, rejects):
    for line_number, text in enumerate(rows, start=1):
        try:
            score = int(text.strip())
        except ValueError:
            rejects.append((line_number, "number"))
            continue
        if not 0 <= score <= 100:
            rejects.append((line_number, "range"))
            continue
        yield score

for rows in [["90", "oops", "60", "101"], ["0", "59", "100"], []]:
    rejects = []
    passes = sum(score >= 60 for score in valid_scores(iter(rows), rejects))
    print(f"passes={passes} rejects={rejects}")

# The source is traversed once. For very large reject volumes, replace list append with a report writer/callback rather than promising constant memory for the whole pipeline.`
                },
                interview: {question: "Can a generator be traversed again after sum consumes it?", answer: "No; an exhausted generator stays exhausted. Construct a new generator from a reusable source or materialize deliberately if multiple traversals are required. Laziness saves result storage but does not eliminate storage used by the source or reject report."},
                readiness: ["I can show when yielding and rejection recording happen, including empty input.", "I can process a one-shot iterator without a second pass or silently dropping malformed rows."]
            },
            "Object-Oriented Python and Design Patterns": {
                prerequisites: "Python Modules 1–2 and dictionaries. Understand a class instance, method calls, and why each object should own its mutable data.",
                setup: "Python 3.11+, save as library.py and run python library.py. The solution uses only dataclasses and typing from the standard library. Protocol describes an interface for static tools; it is not a runtime validation mechanism.",
                workedExample: {
                    problem: "Issue a book through a service that depends on a repository contract, not a file format.",
                    steps: [
                        "Create a frozen Book('B1', 'Loops') value and an in-memory repository containing it. The service asks repo.get('B1') rather than opening a file itself.",
                        "Issue B1: verify existence and absence from the service's active-loan set, then add B1 and return issued.",
                        "Issue B1 again: return unavailable without adding a duplicate loan. A lookup for B9 returns not found. Keep file persistence as a separate adapter, not inside the Book class."
                    ],
                    expected: "The three calls return issued, unavailable, not found. Only B1 is recorded as on loan."
                },
                practice: {
                    task: "Implement Book, a Books protocol, MemoryBooks, and Library.issue(book_id) with exactly the outcomes above. Each Library must own its own loan set; demonstrate missing and duplicate issues.",
                    cases: [
                        {input: "issue('B1'), issue('B1') with B1 present", expected: "issued, unavailable"},
                        {input: "issue('B9') with only B1 present", expected: "not found"},
                        {input: "first issue in two separately created Library instances", expected: "issued in both; loan sets are not shared"}
                    ],
                    hint: "Put self.on_loan = set() in __init__, not as a mutable class attribute. The service can use any object implementing get; no repository inheritance is required.",
                    solution: `from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class Book:
    book_id: str
    title: str

class Books(Protocol):
    def get(self, book_id: str) -> Book | None: ...

class MemoryBooks:
    def __init__(self, books):
        self._books = {book.book_id: book for book in books}
    def get(self, book_id):
        return self._books.get(book_id)

class Library:
    def __init__(self, books: Books):
        self.books = books
        self.on_loan = set()
    def issue(self, book_id):
        if self.books.get(book_id) is None:
            return "not found"
        if book_id in self.on_loan:
            return "unavailable"
        self.on_loan.add(book_id)
        return "issued"

repo = MemoryBooks([Book("B1", "Loops")])
library = Library(repo)
print(library.issue("B1"))
print(library.issue("B1"))
print(library.issue("B9"))
print(Library(repo).issue("B1"))

# The fake repository is composed into the service. Loans are intentionally session-local here; persistent, shared loans require a transactional repository in the later capstone.`
                },
                interview: {question: "How does a Protocol differ from an abstract base class?", answer: "A Protocol lets a static checker accept objects with matching members without explicit inheritance. An ABC can require methods through nominal inheritance and prevent instantiation when abstract methods are missing. Neither automatically makes a repository durable or thread-safe."},
                readiness: ["I can swap a fake repository without changing Library.issue.", "I can explain why frozen dataclasses are shallowly frozen and why loan state must not be a shared class attribute."]
            },
            "Error Handling, Context, and Production Practices": {
                prerequisites: "Python Modules 1–3, exception flow, file reading, and simple classes. Understand that an error message for a user and a diagnostic for an operator can differ.",
                setup: "Python 3.11+. Save solution as config_check.py and run python config_check.py. It uses io.StringIO to test resource cleanup without creating files or reading real environment secrets.",
                workedExample: {
                    problem: "Read a positive batch-size setting while guaranteeing cleanup on normal and invalid paths.",
                    steps: [
                        "Call load_limit on a text stream containing ' 25 '. Enter with stream as handle, read its text, and convert to int.",
                        "If the value is outside 1–1000, raise ConfigError('LIMIT must be 1..1000'). If conversion fails, raise ConfigError('LIMIT must be an integer') from the original ValueError.",
                        "Leave the with block on success or failure. In both paths the stream is closed. Let an outer boundary display the domain message; never log raw secrets."
                    ],
                    expected: "Input '25' returns 25 and closes the stream. Input 'oops' raises ConfigError with a ValueError cause and also closes the stream."
                },
                practice: {
                    task: "Implement load_limit(stream) for integer text in 1..1000. It takes ownership of the stream and closes it. Preserve the conversion cause; distinguish conversion failure from a range violation. Demonstrate both using a fresh StringIO per call.",
                    cases: [
                        {input: "StringIO('25')", expected: "value 25; closed=True"},
                        {input: "StringIO('0')", expected: "ConfigError: LIMIT must be 1..1000; closed=True"},
                        {input: "StringIO('oops')", expected: "ConfigError: LIMIT must be an integer; cause is ValueError; closed=True"}
                    ],
                    hint: "The cleanup belongs in with, not in success-only code. Catch ValueError only around int conversion; otherwise you may accidentally relabel your own validation error.",
                    solution: `from io import StringIO

class ConfigError(Exception):
    pass

def load_limit(stream):
    with stream as handle:
        text = handle.read()
        try:
            limit = int(text.strip())
        except ValueError as exc:
            raise ConfigError("LIMIT must be an integer") from exc
        if not 1 <= limit <= 1000:
            raise ConfigError("LIMIT must be 1..1000")
        return limit

for text in ["25", "0", "oops"]:
    stream = StringIO(text)
    try:
        print(load_limit(stream))
    except ConfigError as exc:
        print(f"ConfigError: {exc}")
    print(f"closed={stream.closed}")

# I/O failures are not converted into validation errors: they propagate to a separate operational boundary. This avoids hiding a permission or disk failure behind a misleading integer message.`
                },
                interview: {question: "Why use raise ConfigError(...) from exc?", answer: "It adds a domain-specific message while preserving the original cause and traceback for diagnostics. A broad except Exception followed by pass would hide the failure. Type hints cannot replace this runtime validation of external text."},
                readiness: ["I can demonstrate closure after both conversion and range failure.", "I can preserve exception causes and describe which boundary handles input errors versus I/O failures."]
            },
            "Concurrency, APIs, Testing, and Performance": {
                prerequisites: "Python Modules 1–4, functions that raise exceptions, and a basic understanding of await. The first concurrency test uses a fake dependency, not the public internet.",
                setup: "Python 3.11+, save as retry_demo.py and run python retry_demo.py. asyncio and unittest are in the standard library. The caller supplies a cooperative asynchronous operation; a real HTTP library must have its own timeout and cancellation support.",
                workedExample: {
                    problem: "Retry a timed-out read once but do not retry a permanent validation failure.",
                    steps: [
                        "The fake operation raises TimeoutError on call 1 and returns {'ok': True} on call 2. max_attempts=2 means two total calls, not two extra retries.",
                        "Wrap each await in asyncio.wait_for with a positive timeout. Catch only TimeoutError, sleep between attempts, and re-raise on exhaustion.",
                        "A ValueError must propagate after one call; cancellation must also propagate. The tests assert call counts so a returned success cannot hide excessive retries."
                    ],
                    expected: "The timeout-then-success test observes two calls; the permanent-error test observes one. The exhaustion test observes exactly two and raises TimeoutError."
                },
                practice: {
                    task: "Implement retry_read(operation, max_attempts=2, timeout=0.1). Validate positive settings, retry only TimeoutError, cap attempts, and keep cancellation unhandled. Add deterministic async unit tests without real sleeps or network access.",
                    cases: [
                        {input: "fake raises TimeoutError, then returns {'ok': True}", expected: "{'ok': True}; calls=2"},
                        {input: "fake raises ValueError", expected: "ValueError; calls=1"},
                        {input: "fake always raises TimeoutError; max_attempts=2", expected: "TimeoutError; calls=2"}
                    ],
                    hint: "Use AsyncMock with side_effect and IsolatedAsyncioTestCase. Set retry delay to zero in this learning version; add capped backoff/jitter and idempotency checks for real integrations.",
                    solution: `import asyncio
import unittest
from unittest.mock import AsyncMock

async def retry_read(operation, max_attempts=2, timeout=0.1):
    if max_attempts < 1 or timeout <= 0:
        raise ValueError("positive attempts and timeout required")
    for attempt in range(max_attempts):
        try:
            return await asyncio.wait_for(operation(), timeout)
        except TimeoutError:
            if attempt + 1 == max_attempts:
                raise
            await asyncio.sleep(0)

class RetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_after_timeout(self):
        op = AsyncMock(side_effect=[TimeoutError(), {"ok": True}])
        self.assertEqual(await retry_read(op), {"ok": True})
        self.assertEqual(op.await_count, 2)
    async def test_permanent_error(self):
        op = AsyncMock(side_effect=ValueError("bad response"))
        with self.assertRaises(ValueError):
            await retry_read(op)
        self.assertEqual(op.await_count, 1)
    async def test_exhaustion(self):
        op = AsyncMock(side_effect=TimeoutError())
        with self.assertRaises(TimeoutError):
            await retry_read(op)
        self.assertEqual(op.await_count, 2)

if __name__ == "__main__":
    unittest.main()

# This is a read-only retry contract. Blindly retrying writes can duplicate side effects. wait_for requests cancellation on timeout and may wait for cleanup; it cannot forcibly stop blocking code or an operation that ignores cancellation.`
                },
                interview: {question: "Why doesn't asyncio make a CPU-heavy loop parallel?", answer: "Coroutines share an event-loop thread and cooperate at await points. A CPU loop prevents other tasks from progressing. On GIL-enabled CPython, use processes for parallel pure-Python CPU work, or native code that releases the GIL; benchmark overhead before choosing."},
                readiness: ["I can test success, permanent failure, exhaustion, and cancellation without a live API.", "I can explain total attempts, per-attempt timeout, backoff, and the limits of cooperative cancellation."]
            },
            "Production Python Capstone": {
                prerequisites: "Python Modules 1–5, repository/service separation, targeted errors, unit tests, and safe resource handling. Choose the Inventory and Reorder Platform option for this concrete capstone.",
                setup: "Use Python 3.11+ with sqlite3, argparse, logging, and unittest from the standard library. Build app/service/domain/repository modules in your own capstone folder. Use a disposable local inventory.db for integration tests and document python -m unittest discover as the test command.",
                workedExample: {
                    problem: "Reserve two units of SKU P1 from stock 5 and make duplicate requests harmless.",
                    steps: [
                        "Create items(sku PRIMARY KEY, stock INTEGER CHECK(stock >= 0)) and reservations(request_id PRIMARY KEY, sku, quantity). Seed P1 with 5 units.",
                        "For request R1, validate integer quantity >0, start a transaction, check for an existing request/payload, conditionally decrement stock where stock >= quantity, and insert the reservation.",
                        "Commit both changes together. Replay R1 with the same payload as a no-op success; reject reuse with a different quantity. Inject an insert failure to verify the decrement rolls back."
                    ],
                    expected: "R1(P1,2) leaves stock 3 and one reservation. An identical replay leaves stock 3. A fresh request for 4 is rejected without mutation."
                },
                practice: {
                    task: "Deliver reserve and stock CLI commands backed by SQLite. Acceptance: durable idempotency; no negative stock; atomic reserve; restart persistence; explicit unknown-SKU and insufficient-stock messages; >=12 tests with >=8 automated domain/integration cases; forced-write failure and two-connection contention tests. Include logs without payload secrets, a storage ADR, README commands, and a scripted demo.",
                    cases: [
                        {input: "seed P1=5; reserve R1 P1 2; repeat R1 P1 2; restart; stock P1", expected: "3; exactly one reservation persisted"},
                        {input: "P1=3; reserve R2 P1 4", expected: "insufficient stock; P1=3; no R2"},
                        {input: "seed P1=5; force reservation insert failure after decrement", expected: "rollback; P1=5; no reservation"}
                    ],
                    hint: "Validate quantity before SQL. An UPDATE with a stock >= quantity predicate avoids a stale read-then-write check. Manage a connection explicitly; sqlite3 connection context management handles commit/rollback but does not itself close the connection.",
                    solution: "Implementation route: define reserve(request_id, sku, quantity) at the service boundary and validate nonempty IDs and type(quantity) is int with quantity >0. Open a connection, enable a documented busy timeout, and BEGIN IMMEDIATE for the reserve transaction. Load any existing reservation: if all payload fields match return its stored result, otherwise reject request-ID reuse. Execute parameterized UPDATE items SET stock=stock-? WHERE sku=? AND stock>=? and require rowcount=1; distinguish unknown SKU from insufficient stock within the transaction. Insert the reservation and commit; on every exception roll back, then close in finally. Persist request identity and stock in the same transaction. Test using separate on-disk connections, including an injected exception between update and insert. Do not confuse the in-memory fake's passing tests with SQLite locking evidence. Surface lock exhaustion as an operational error, not insufficient stock."
                },
                interview: {question: "Why is validate-before-write insufficient for concurrent stock changes?", answer: "Two callers can both read the same stock before either writes. A transaction plus a conditional update enforces the rule at storage time; a persisted unique request ID prevents replay duplication. Validation is still needed for syntax, types, and clear error messages."},
                readiness: ["I can reproduce reserve, replay, rejection, restart, and rollback with a fresh fixture.", "I can explain transaction/locking assumptions and show which tests use real SQLite rather than a fake."]
            },
            "Career Specialization Tracks (Optional)": {
                prerequisites: "Complete Python Modules 1–6. Choose one track, not all three at once. The dependency-free automation option below turns a portfolio claim into a small, inspectable deliverable.",
                setup: "Use Python 3.11+ and pathlib/unittest only for the initial automation artifact. Work inside a dedicated user-created learning folder with sample files, never your real documents. First build a pure planner; no files should move during planning.",
                workedExample: {
                    problem: "Plan image renames reproducibly without overwriting a file or changing anything in dry-run.",
                    steps: [
                        "Given names ['b.JPG', 'a.jpg', 'notes.txt'], select .jpg names case-insensitively and sort by (casefolded name, original name). This fixes ordering across runs.",
                        "Assign destinations image-001.jpg and image-002.jpg in that order. Before emitting a plan, reject any destination already present in the complete input name set, including case-insensitive collisions.",
                        "Print the proposed moves in dry-run and compare the directory before/after. Only a separate opt-in apply command may rename files; if any destination exists, stop before the first rename."
                    ],
                    expected: "a.jpg -> image-001.jpg; b.JPG -> image-002.jpg. notes.txt is untouched. The pure planner does not perform filesystem I/O."
                },
                practice: {
                    task: "Implement plan(names) for those exact JPG rules and collision checks. Use it as the first tested component of the automation portfolio track. Artifact acceptance: dry-run default, explicit apply option, preflight validation, success journal and documented rollback limits, three collision/ordering tests, README demo. Backend alternative: implement one tested CRUD resource with explicit status codes. Data alternative: publish a deterministic cleaned dataset, missing-value report and train/test split with no target leakage. Pick only one extension.",
                    cases: [
                        {input: "['b.JPG', 'a.jpg', 'notes.txt']", expected: "[('a.jpg', 'image-001.jpg'), ('b.JPG', 'image-002.jpg')]"},
                        {input: "['a.jpg', 'IMAGE-001.JPG']", expected: "ValueError: destination exists; no moves"},
                        {input: "['notes.txt']", expected: "[]"}
                    ],
                    hint: "Planning and applying are different responsibilities. For safety, reject rather than attempting clever renaming cycles when a generated destination is already occupied.",
                    solution: `from pathlib import Path

def plan(names):
    occupied = {name.casefold() for name in names}
    photos = sorted(
        (name for name in names if Path(name).suffix.casefold() == ".jpg"),
        key=lambda name: (name.casefold(), name),
    )
    moves = [(name, f"image-{index:03d}.jpg")
             for index, name in enumerate(photos, start=1)]
    if any(target.casefold() in occupied for _, target in moves):
        raise ValueError("destination exists")
    return moves

print(plan(["b.JPG", "a.jpg", "notes.txt"]))
print(plan(["notes.txt"]))
try:
    plan(["a.jpg", "IMAGE-001.JPG"])
except ValueError as exc:
    print(f"ValueError: {exc}")

# For the opt-in extension, preflight every source and target, journal each completed rename, and reverse only journaled moves whose current state still matches. A preflight alone cannot prevent external concurrent changes; document the single-writer assumption and partial-failure recovery instead of promising universal atomic rollback.`
                },
                interview: {question: "How would you demonstrate depth rather than listing three frameworks?", answer: "Show a reproducible artifact, its acceptance tests and failure handling. For this automation track, demonstrate a dry-run with zero changes and a collision rejected before any move. Set a 60-day plan with measurable milestones: tested planner, safe apply/journal, documented recovery demo, then one reviewed release."},
                readiness: ["I can demonstrate the chosen artifact's normal and failure paths and state its safety limits.", "I can show one role-focused deliverable and a dated roadmap with testable milestones rather than claiming mastery of every track."]
            }
        },
        csharpProgramming: {
            "C# and .NET Foundations": {
                prerequisites: "None. A terminal runs commands; an SDK supplies the compiler and runtime. Begin with visible console output before CLR or IL details.",
                setup: "Install a supported .NET SDK, version 8 or later. Verify dotnet --list-sdks. Run dotnet new console -n ScoreDemo, then cd ScoreDemo. Replace all of Program.cs with the solution and run dotnet run -- 75. The -- separates program arguments from dotnet options. No additional packages.",
                workedExample: {
                    problem: "Read one score from a command argument and print Pass at 60 or above, otherwise Retry.",
                    steps: [
                        "First replace Program.cs with Console.WriteLine(\"Hello, C#!\"); and run dotnet run. Confirm the greeting; this top-level form has an entry point generated by the compiler.",
                        "Run dotnet run -- 75. args[0] contains the text \"75\". Check args.Length first, then use int.TryParse to get an int without throwing on bad numeric text.",
                        "Require 0 <= score <= 100. Only after validation compare score >= 60. For 75 print Pass; for 59 print Retry. An out-of-range number is not a failing grade: it is invalid input."
                    ],
                    expected: "75 -> Pass; 59 -> Retry; hello or 101 -> Invalid score. Each invocation prints exactly one result line."
                },
                practice: {
                    task: "Implement the exact one-argument pass checker above. Reject missing/extra arguments, non-integer text, and numbers outside 0–100. Print only Pass, Retry, or Invalid score.",
                    cases: [
                        {input: "dotnet run -- 60", expected: "Pass"},
                        {input: "dotnet run -- 59", expected: "Retry"},
                        {input: "dotnet run -- hello", expected: "Invalid score"},
                        {input: "dotnet run -- 101", expected: "Invalid score"}
                    ],
                    hint: "Combine the length, TryParse, and range guards using ||. Short-circuiting prevents args[0] access when the length is wrong.",
                    solution: `using System;

if (args.Length != 1 ||
    !int.TryParse(args[0], out int score) ||
    score < 0 || score > 100)
{
    Console.WriteLine("Invalid score");
    return;
}
Console.WriteLine(score >= 60 ? "Pass" : "Retry");

// TryParse returns a success flag and assigns the out variable. The range guard is still necessary: successfully parsing 101 does not make it a valid score.`
                },
                interview: {question: "How are a value copy and a reference copy different?", answer: "Assigning an int copies its value, so changing the new variable does not change the original. Assigning a List<int> variable copies a reference; both variables then refer to the same mutable list. This is about assignment semantics, not a rule that every value type lives on the stack."},
                readiness: ["I can create a project, replace Program.cs, and pass arguments without mixing template entry-point styles.", "I can trace short-circuit validation and explain why parsing success and domain validity differ."]
            },
            "Object-Oriented C# and Design Principles": {
                prerequisites: "C# Module 1: methods, constructor syntax, boolean checks, and basic exceptions. A property exposes controlled access to state; an interface names behavior a caller can depend on.",
                setup: "Use a .NET 8+ console project. Replace Program.cs with the complete solution; dotnet run. Keep top-level statements before the type declarations. The exercise is synchronous deliberately; async notification is introduced in Module 4.",
                workedExample: {
                    problem: "Borrow a book once and notify a caller without tying the library rule to email.",
                    steps: [
                        "Book('B1') starts with Available=true. Its private setter prevents callers from directly marking availability.",
                        "Library depends on INotifier. Borrow checks Available, changes it to false, and sends a message through that interface.",
                        "A second Borrow returns false before sending. A console or test notifier can implement the same interface without changing the borrowing rule."
                    ],
                    expected: "First call prints Borrowed B1 and returns true. Second returns false with no second notification. Final Available is false."
                },
                practice: {
                    task: "Implement the synchronous Book, INotifier, and Library shown below. An empty/whitespace book ID must be rejected at construction. Notify once per successful borrow, never for rejection. Use an in-memory counting notifier to verify behavior.",
                    cases: [
                        {input: "Book B1; Borrow twice", expected: "True, False; notifications=1; Available=False"},
                        {input: "Book with whitespace ID", expected: "ArgumentException; no book created"},
                        {input: "Two separate books B1 and B2, borrow each once", expected: "both True; notifications=2"}
                    ],
                    hint: "Constructor validation protects initial state. Put the duplicate-borrow guard before state mutation and notification. This learning notifier cannot fail; durable external notification requires a separate reliability design.",
                    solution: `using System;

var notifier = new CountingNotifier();
var library = new Library(notifier);
var book = new Book("B1");
Console.WriteLine(library.Borrow(book));
Console.WriteLine(library.Borrow(book));
Console.WriteLine($"notifications={notifier.Count}; Available={book.Available}");

public sealed class Book
{
    public string Id { get; }
    public bool Available { get; private set; } = true;
    public Book(string id)
    {
        if (string.IsNullOrWhiteSpace(id)) throw new ArgumentException("ID required");
        Id = id;
    }
    public bool TryBorrow()
    {
        if (!Available) return false;
        Available = false;
        return true;
    }
}
public interface INotifier { void Send(string message); }
public sealed class CountingNotifier : INotifier
{
    public int Count { get; private set; }
    public void Send(string message) { Count++; Console.WriteLine(message); }
}
public sealed class Library(INotifier notifier)
{
    public bool Borrow(Book book)
    {
        if (!book.TryBorrow()) return false;
        notifier.Send($"Borrowed {book.Id}");
        return true;
    }
}
// The constructor-injected collaborator is composed, not inherited. Book owns availability; Library coordinates notification only after a successful transition.`
                },
                interview: {question: "Would a record automatically be the best Book type?", answer: "Not necessarily. This Book has identity and changing availability, so a class is clear. Records are useful for value-centric request/response data, but record syntax does not guarantee deep immutability. Choose by equality and state semantics, not brevity alone."},
                readiness: ["I can use a fake notifier and prove one notification for two borrow attempts.", "I can explain entity identity, constructor invariants, and why the interface belongs at a real substitution boundary."]
            },
            "Collections, Generics, LINQ, and Exceptions": {
                prerequisites: "C# Modules 1–2, List<T>, method calls, lambdas such as x => x.Score, and the difference between a sequence and a lookup.",
                setup: "Use a .NET 8+ console project; replace Program.cs and run dotnet run. Explicit using directives in the solution avoid depending on implicit project imports. No database or packages.",
                workedExample: {
                    problem: "Build a case-insensitive student index and a deterministic top-scorer list.",
                    steps: [
                        "Insert S1:80/Asha into Dictionary<string,Student>(StringComparer.OrdinalIgnoreCase). A later s1:99 is a duplicate of S1, not a new ID; keep the first entry.",
                        "Reject invalid scores before adding them. Query the stored values with Where(score >=85), then OrderByDescending(score) and ThenBy(Id, OrdinalIgnoreCase) for ties.",
                        "Materialize with ToList at the boundary. This captures the report once; repeating enumeration of an unmaterialized query could observe source changes."
                    ],
                    expected: "For S1:80, s1:99, S2:90, S3:90: one duplicate reject and ranked IDs S2,S3."
                },
                practice: {
                    task: "Write Report over Student records: case-insensitive unique IDs (first valid wins), scores 0–100 only, top threshold >=85, ties by case-insensitive ID. Return a concrete list of IDs and count all invalid-score or duplicate records as rejects.",
                    cases: [
                        {input: "S1:80, s1:99, S2:90, S3:90", expected: "top=S2,S3; rejects=1"},
                        {input: "S1:101, s1:85", expected: "top=s1; rejects=1"},
                        {input: "empty array", expected: "top=; rejects=0"}
                    ],
                    hint: "Range-check first, then TryAdd; an invalid score must not reserve its ID. Do not rely on dictionary traversal order for a ranked report.",
                    solution: `using System;
using System.Collections.Generic;
using System.Linq;

Report(new[] { new Student("S1", 80), new Student("s1", 99),
    new Student("S2", 90), new Student("S3", 90) });
Report(new[] { new Student("S1", 101), new Student("s1", 85) });
Report(Array.Empty<Student>());

static void Report(IEnumerable<Student> rows)
{
    var byId = new Dictionary<string, Student>(StringComparer.OrdinalIgnoreCase);
    int rejects = 0;
    foreach (var student in rows)
    {
        if (student.Score < 0 || student.Score > 100 ||
            !byId.TryAdd(student.Id, student)) rejects++;
    }
    var top = byId.Values.Where(s => s.Score >= 85)
        .OrderByDescending(s => s.Score)
        .ThenBy(s => s.Id, StringComparer.OrdinalIgnoreCase)
        .Select(s => s.Id).ToList();
    Console.WriteLine($"top={string.Join(",", top)}; rejects={rejects}");
}
public record Student(string Id, int Score);

// The exercise assumes nonempty IDs supplied by its caller. An API/import boundary must additionally validate IDs. Short-circuit || prevents TryAdd for an invalid score.`
                },
                interview: {question: "Does ToList make the contained Student objects deeply immutable?", answer: "No. It snapshots the sequence into a new list; it does not recursively clone elements. Here positional record properties are init-only, but mutable reference-valued properties would still be shared. Also distinguish LINQ-to-Objects from queries translated by a database provider."},
                readiness: ["I can predict the effect of duplicate IDs with different casing and invalid-first rows.", "I can explain deferred execution, materialization, and deterministic tie handling without relying on dictionary order."]
            },
            "Async, Concurrency, and Resilience": {
                prerequisites: "C# Modules 1–3, exceptions, method return values, and basic Task/await vocabulary. Cancellation is a cooperative signal, not forced termination.",
                setup: "Use a .NET 8+ console project. Replace Program.cs with the solution and run dotnet run. Task.Delay simulates cooperative I/O; no HTTP calls, packages, or wall-clock performance assertions are needed.",
                workedExample: {
                    problem: "Combine two asynchronous reads in input order and handle a caller-cancelled operation.",
                    steps: [
                        "Start ReadAsync('A', token) and ReadAsync('B', token) before awaiting either. Each waits asynchronously using Task.Delay; neither needs Task.Run for this I/O simulation.",
                        "Await Task.WhenAll(first, second). Its result array follows task input order even if B completes first, so joining the strings yields AB.",
                        "For a deterministic cancellation test, cancel a CancellationTokenSource before calling the same pipeline. Both delays observe the token; catch OperationCanceledException at the caller boundary."
                    ],
                    expected: "Normal run prints AB. A pre-cancelled run prints cancelled. No .Result or .Wait is used."
                },
                practice: {
                    task: "Implement LoadAsync(token) that returns AB from two concurrently started ReadAsync calls and propagates caller cancellation. Keep workers free of shared mutable state. In main demonstrate a successful and pre-cancelled run.",
                    cases: [
                        {input: "LoadAsync(CancellationToken.None)", expected: "AB"},
                        {input: "LoadAsync(token already cancelled)", expected: "OperationCanceledException; boundary prints cancelled"}
                    ],
                    hint: "Pass the token into the actual awaited I/O, not just the outer method. Task.WhenAll is not a concurrency limiter for an arbitrarily large input.",
                    solution: `using System;
using System.Threading;
using System.Threading.Tasks;

Console.WriteLine(await LoadAsync(CancellationToken.None));
using var stop = new CancellationTokenSource();
stop.Cancel();
try { Console.WriteLine(await LoadAsync(stop.Token)); }
catch (OperationCanceledException) when (stop.IsCancellationRequested)
{
    Console.WriteLine("cancelled");
}

static async Task<string> ReadAsync(string value, CancellationToken token)
{
    await Task.Delay(value == "A" ? 20 : 1, token);
    return value;
}
static async Task<string> LoadAsync(CancellationToken token)
{
    Task<string> first = ReadAsync("A", token);
    Task<string> second = ReadAsync("B", token);
    string[] values = await Task.WhenAll(first, second);
    return string.Concat(values);
}

// Input-order aggregation avoids completion-order output. For a large feed, bound in-flight work with a semaphore or bounded channel. Do not retry caller cancellation as a transient failure.`
                },
                interview: {question: "Does await move work to a new thread?", answer: "No. The method initially runs synchronously. An incomplete await returns control to its caller and arranges continuation after completion. True asynchronous I/O can avoid blocking a thread while waiting; CPU work still needs an appropriate execution strategy."},
                readiness: ["I can show successful and cancelled behavior without relying on a timing race.", "I can distinguish task composition from thread creation, concurrency limiting, and retry policy."]
            },
            "ASP.NET Core, Data Access, and Architecture": {
                prerequisites: "C# Modules 1–4 and basic HTTP: GET reads a resource; POST creates one; a status code explains the outcome. Learn endpoint validation before connecting EF Core.",
                setup: "Use a .NET 8+ SDK. Run dotnet new web -n TaskApi then cd TaskApi. Replace Program.cs with the solution. Run dotnet run --urls http://localhost:5050. Use an HTTP client on this local address only. This baseline uses in-memory storage, intentionally no EF provider or external packages.",
                workedExample: {
                    problem: "Create a task with a valid title and retrieve it using its returned location.",
                    steps: [
                        "POST /tasks with Content-Type: application/json and body {\"title\":\"Read docs\"}. The endpoint trims the title and requires 1–100 characters.",
                        "Assign an ID, store the response DTO, and return 201 Created with Location /tasks/1 and the stored JSON. Follow the Location using GET.",
                        "POST {\"title\":\"  \"} returns 400 without storing anything. GET /tasks/999 returns 404. Validation is explicit here; a minimal API does not inherit controller validation behavior by assumption."
                    ],
                    expected: "On a fresh process, valid POST returns 201 and {\"id\":1,\"title\":\"Read docs\"}; GET /tasks/1 returns 200 with the same object. Invalid title ->400; missing task ->404."
                },
                practice: {
                    task: "Implement these create/get endpoints with DTOs and explicit validation. Acceptance: 201 with Location; trimming; title length 1–100; 400 invalid; 404 missing; safe concurrent ID assignment. Prove that data disappears after restart, then document the next EF Core migration/repository step rather than claiming this dictionary is persistence.",
                    cases: [
                        {input: "Fresh process; POST /tasks {\"title\":\" Read docs \"}; GET returned Location", expected: "201 then 200; title is Read docs; id is 1"},
                        {input: "POST /tasks {\"title\":\"  \"}", expected: "400; title length error; no task added"},
                        {input: "GET /tasks/999", expected: "404"}
                    ],
                    hint: "Keep request/response DTOs separate. Use Interlocked for ID generation and ConcurrentDictionary for per-key storage; these do not make a multi-record business transaction atomic.",
                    solution: `using System.Collections.Concurrent;
using System.Threading;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();
var tasks = new ConcurrentDictionary<int, TaskDto>();
int nextId = 0;

app.MapPost("/tasks", (CreateTask request) =>
{
    string title = (request.Title ?? "").Trim();
    if (title.Length is < 1 or > 100)
        return Results.BadRequest(new { error = "Title must be 1..100 characters" });
    int id = Interlocked.Increment(ref nextId);
    var task = new TaskDto(id, title);
    tasks[id] = task;
    return Results.Created($"/tasks/{id}", task);
});
app.MapGet("/tasks/{id:int}", (int id) =>
    tasks.TryGetValue(id, out var task) ? Results.Ok(task) : Results.NotFound());
app.Run();

public record CreateTask(string? Title);
public record TaskDto(int Id, string Title);

// Malformed JSON is rejected by request binding before the handler. For the persistence extension, register a scoped DbContext, add a real provider/migration, and move storage behind a service boundary; never share one DbContext across concurrent requests or singleton services.`
                },
                interview: {question: "Why is injecting a scoped DbContext into a singleton service a problem?", answer: "The singleton outlives request scopes and may retain/use the context concurrently. DbContext is not thread-safe. Keep request work scoped, or explicitly create and dispose a scope/context per background unit of work. A repository wrapper does not fix an incorrect lifetime."},
                readiness: ["I can demonstrate the three HTTP outcomes and distinguish binding, validation, and storage failures.", "I can explain why this baseline is not durable or authenticated and outline a migration with real database tests."]
            },
            "Production C# Capstone and Interview Readiness": {
                prerequisites: "C# Modules 1–5: API contracts, async cancellation, DI lifetimes, persistence/migrations, and tests. Choose the Task Workflow API option for a concrete delivery scope.",
                setup: "Use a supported .NET 8+ SDK, ASP.NET Core, EF Core with a local relational provider, and an existing chosen test framework. Pin compatible package versions in the capstone, document migration and dotnet test commands, and use an isolated database for integration tests. Do not expose the learning API publicly without an authentication design.",
                workedExample: {
                    problem: "Prevent two clients from silently overwriting the same task edit.",
                    steps: [
                        "Create task 1 with title Draft and version 1. GET returns the version; clients A and B both read it.",
                        "Client A sends PUT with expectedVersion=1 and title Reviewed. Update using an atomic id/version predicate and increment to version 2.",
                        "Client B sends a stale PUT with expectedVersion=1. Zero updated rows means conflict, not success. Return 409; do not overwrite Reviewed. GET must still show version 2."
                    ],
                    expected: "A receives 200 with Reviewed/version 2; B receives 409. Restarting the API preserves Reviewed/version 2."
                },
                practice: {
                    task: "Deliver create/get/update for tasks with explicit optimistic concurrency. Acceptance: 201 create; 200 read/update; 400 empty or >100-character title; 404 unknown ID; 409 stale version; durable state after restart; cancellation propagated; errors do not expose SQL/internal stack traces. Include >=12 automated tests, real-database concurrency and rollback checks, migration instructions, structured operation logs, one ADR, and a reproducible demo.",
                    cases: [
                        {input: "Task 1 Draft/v1; PUT Reviewed with expectedVersion 1; stale PUT Other/v1", expected: "200 Reviewed/v2, then 409; stored title remains Reviewed"},
                        {input: "PUT existing task with whitespace title and current version", expected: "400; title and version unchanged"},
                        {input: "GET /tasks/999", expected: "404, not 500 or an empty success object"}
                    ],
                    hint: "Comparing versions only in service memory is racy. Enforce the version predicate in the database update or use a correctly configured concurrency token.",
                    solution: "Implementation route: use separate CreateTaskRequest, UpdateTaskRequest(title, expectedVersion), and TaskResponse DTOs. Validate title/positive version at the API boundary; keep domain validation reusable. Store an integer Version configured as a concurrency token, or issue an atomic parameterized UPDATE WHERE Id=id AND Version=expectedVersion that sets Title and Version=Version+1. With tracked EF entities, compare the loaded version to the submitted one before mutation and handle DbUpdateConcurrencyException for races after that check. Map missing rows to 404 and stale rows to 409 without automatic overwrite. Scope DbContext per request and await operations with cancellation. Test with two contexts/connections, not a shared in-memory collection. Reopen the database after API restart to prove durability. Add a failure-injection test verifying no partial state change when persistence fails. Document that this exercise uses body versions, not HTTP ETag/If-Match semantics; do not mix 409 with an undocumented 412 policy."
                },
                interview: {question: "What does a passing mocked repository test fail to prove here?", answer: "It cannot prove the database enforces concurrency, migration compatibility, rollback, or durability. Use domain unit tests for decisions and real-provider integration tests for storage behavior. Explain unresolved authorization, capacity, and operational limits before describing the project as production-ready."},
                readiness: ["I can demonstrate a real stale-write conflict and persistence after restart using the documented commands.", "I can show the test matrix, architecture dependency direction, concurrency ADR, and honest deployment limitations."]
            }
        },
        cppProgramming: {
            "C++ Foundations and Compilation Model": {
                prerequisites: "None. Learn to save, compile and run a single source file before headers, linking details, or history. A compiler turns source into a program; editing source alone does not update the executable.",
                setup: "Use a C++20-capable compiler. On Windows, open a Visual Studio Developer PowerShell with C++ Build Tools available. Save the complete solution as main.cpp; run cl /std:c++20 /EHsc /W4 main.cpp /Fe:score.exe then .\\score.exe. With an existing GCC toolchain use g++ -std=c++20 -Wall -Wextra -pedantic main.cpp -o score.exe then .\\score.exe. No libraries beyond the standard library.",
                workedExample: {
                    problem: "Read one score, check it, and print Pass or Retry without reading an uninitialized value.",
                    steps: [
                        "First run the module's hello program and confirm output. Then initialize int score{0}; before reading; braces give a known starting value.",
                        "Use if (!(std::cin >> score)) to detect extraction failure. Do not compare score to a threshold until input succeeds and the range is 0–100.",
                        "For input 60 the range check succeeds and score >= 60 is true, so print Pass. For 59 print Retry. Return a nonzero exit code for invalid input."
                    ],
                    expected: "60 -> Pass; 59 -> Retry; hello or 101 -> Invalid score. The solution reads one integer token; strict whole-line parsing is a later extension."
                },
                practice: {
                    task: "Implement the single-token score checker below. Accept an integer token in 0–100 inclusive, print Pass for >=60 and Retry otherwise. If extraction fails or range is invalid, print Invalid score and exit 1. Ignore remaining input tokens in this introductory version.",
                    cases: [
                        {input: "60", expected: "Pass; exit 0"},
                        {input: "59", expected: "Retry; exit 0"},
                        {input: "hello", expected: "Invalid score; exit 1"},
                        {input: "101", expected: "Invalid score; exit 1"}
                    ],
                    hint: "Use || to combine input failure and range guards. std::cin >> int reads a numeric prefix, so do not claim that this validates an entire line such as 75oops.",
                    solution: `#include <iostream>

int main() {
    int score{0};
    if (!(std::cin >> score) || score < 0 || score > 100) {
        std::cout << "Invalid score" << std::endl;
        return 1;
    }
    std::cout << (score >= 60 ? "Pass" : "Retry") << std::endl;
    return 0;
}
// Initialization prevents indeterminate reads, and the guard prevents invalid data reaching classification. Rebuild before rerunning each edit. A full-line validator can later use getline plus from_chars with an end-pointer check.`
                },
                interview: {question: "How does a compiler error differ from a linker error?", answer: "A compiler error can report invalid syntax or types in a translation unit. A linker error can report a declared/called function with no linked definition. Neither is fixed by adding runtime input validation; fix the first relevant build diagnostic."},
                readiness: ["I can save, compile, change, rebuild, and execute main.cpp using one toolchain.", "I can trace input failure and boundary values, and state the difference between token and full-line validation."]
            },
            "Pointers, Ownership, and Lifetime": {
                prerequisites: "C++ Module 1, functions, scope, references, and value initialization. An owner controls lifetime; a borrower may use an object only while it remains alive.",
                setup: "Use C++20 and the same main.cpp build command from Module 1. The solution needs only iostream, memory, and utility. Compile with warnings enabled; no manual new/delete or external tools are necessary for this first ownership exercise.",
                workedExample: {
                    problem: "Transfer exclusive ownership of a resource while leaving a safe observing reference within its valid lifetime.",
                    steps: [
                        "Construct auto first = std::make_unique<Resource>(42). first owns the resource; read_id(*first) borrows it for one function call without taking ownership.",
                        "auto second = std::move(first) moves the unique_ptr. first becomes null and second owns the same Resource; check first before dereferencing.",
                        "Leave second's scope. Its destructor destroys Resource exactly once. Never return or retain a pointer/reference that will outlive this owner."
                    ],
                    expected: "The solution prints 42, empty, 42, released 42 on separate lines. Destruction is visible after the last valid read."
                },
                practice: {
                    task: "Implement Resource with an ID and destructor message, read_id(const Resource&) as a borrower, and an explicit unique_ptr ownership transfer. Verify that the moved-from owner is null and the resource is destroyed once. Repeat with ID 7.",
                    cases: [
                        {input: "Resource ID 42 transferred first->second", expected: "42; empty; 42; released 42 exactly once"},
                        {input: "Resource ID 7 transferred first->second", expected: "7; empty; 7; released 7 exactly once"}
                    ],
                    hint: "std::move does not destroy the Resource; unique_ptr's move constructor transfers its ownership. Do not dereference first after the move or manually delete second.get().",
                    solution: `#include <iostream>
#include <memory>
#include <utility>

struct Resource {
    int id;
    explicit Resource(int value) : id{value} {}
    ~Resource() { std::cout << "released " << id << std::endl; }
};
int read_id(const Resource& resource) { return resource.id; }
void demo(int id) {
    auto first = std::make_unique<Resource>(id);
    std::cout << read_id(*first) << std::endl;
    auto second = std::move(first);
    std::cout << (first ? "owned" : "empty") << std::endl;
    std::cout << read_id(*second) << std::endl;
}
int main() { demo(42); demo(7); }

// Each demo owns one object throughout the transfer. read_id borrows only for the call. A Resource containing only safe standard members normally follows Rule of Zero; this destructor exists solely to make cleanup observable.`
                },
                interview: {question: "Does using shared_ptr automatically prevent leaks and dangling references?", answer: "No. Ownership cycles can leak unless broken with weak_ptr or redesigned. Raw references borrowed from a shared_ptr can still dangle after the last owner disappears. Choose shared ownership only when the domain truly requires shared lifetime."},
                readiness: ["I can draw the owner before and after the move and identify every valid dereference.", "I can show one destructor call and explain why a borrower must not outlive the owner."]
            },
            "STL, Algorithms, and Generic Programming": {
                prerequisites: "C++ Modules 1–2, vector, loops, references, and object lifetimes. begin/end form a half-open range: begin is included, end is not an element.",
                setup: "Use C++20 with warnings. Save the solution as main.cpp and compile with the Module 1 command. Headers are included explicitly. Data is a small in-memory inventory; no files or dependencies.",
                workedExample: {
                    problem: "List low-stock inventory by quantity, then ID, with reproducible tie ordering.",
                    steps: [
                        "Start with B:2, A:2, C:5, D:0. Use copy_if with stock <3 into a new vector: B:2, A:2, D:0. Zero is valid stock.",
                        "Sort using ascending stock, then ascending ID. Comparing tuple keys with < supplies strict weak ordering; using <= would incorrectly say an item precedes itself.",
                        "Traverse the sorted vector to print D:0 A:2 B:2. The original vector stays in input order; no iterators from it are invalidated by editing the separate report."
                    ],
                    expected: "D:0 A:2 B:2. With no low-stock items print (none)."
                },
                practice: {
                    task: "Implement low_stock over vector<Item>, selecting stock <3 and sorting by (stock,id). Assume unique nonempty IDs and nonnegative stock already validated at ingestion. Preserve the input and output (none) for an empty report.",
                    cases: [
                        {input: "B:2, A:2, C:5, D:0", expected: "D:0 A:2 B:2"},
                        {input: "A:3, B:5", expected: "(none)"},
                        {input: "empty inventory", expected: "(none)"}
                    ],
                    hint: "Use back_inserter so copy_if grows the destination safely. A transform into output.begin() would require an already sized destination.",
                    solution: `#include <algorithm>
#include <iostream>
#include <iterator>
#include <string>
#include <tuple>
#include <vector>

struct Item { std::string id; int stock; };
void low_stock(const std::vector<Item>& items) {
    std::vector<Item> report;
    std::copy_if(items.begin(), items.end(), std::back_inserter(report),
        [](const Item& item) { return item.stock < 3; });
    std::sort(report.begin(), report.end(), [](const Item& a, const Item& b) {
        return std::tie(a.stock, a.id) < std::tie(b.stock, b.id);
    });
    if (report.empty()) std::cout << "(none)";
    for (std::size_t i{0}; i < report.size(); ++i) {
        if (i != 0) std::cout << ' ';
        std::cout << report[i].id << ':' << report[i].stock;
    }
    std::cout << std::endl;
}
int main() {
    low_stock({{"B", 2}, {"A", 2}, {"C", 5}, {"D", 0}});
    low_stock({{"A", 3}, {"B", 5}});
    low_stock({});
}
// Filtering is O(n); sorting k selected items is O(k log k). The extra vector costs O(k) storage and keeps the source unchanged.`
                },
                interview: {question: "When does vector::push_back invalidate iterators?", answer: "If it reallocates, all pointers/references/iterators to elements are invalidated. Without reallocation, existing element references remain valid, but end changes. reserve can reduce reallocations; it changes capacity, not size, and is not permission to index unconstructed elements."},
                readiness: ["I can verify the comparator returns false for equal keys and produces deterministic ties.", "I can explain the report's complexity, half-open ranges, and why back_inserter is safe here."]
            },
            "Modern C++: RAII, Move Semantics, and Language Features": {
                prerequisites: "C++ Modules 1–3, ownership and STL algorithms. Learn optional as an explicit maybe-value before mixing variant alternatives or non-owning views into an API.",
                setup: "Use a C++20 toolchain and the Module 1 compile command. Save the complete solution as main.cpp. std::from_chars uses locale-independent integer parsing and needs no exception or external parser.",
                workedExample: {
                    problem: "Replace the ambiguous -1 parse-failure sentinel with optional<int> for a score parser.",
                    steps: [
                        "Parse the text with from_chars using [data, data+size). Check both the error code and that the returned pointer reached the end: '75oops' is not a valid whole input.",
                        "Reject a converted value outside 0–100, returning nullopt. A real score of 0 must remain a successful value.",
                        "At the caller use if (value), then dereference only inside that branch. An engaged optional containing zero is truthy; it does not mean the numeric result is nonzero."
                    ],
                    expected: "'75' -> 75; '0' -> 0; '75oops' -> invalid. This strict parser intentionally rejects surrounding whitespace and leading '+'."
                },
                practice: {
                    task: "Implement parse_score(string_view) returning optional<int>, with complete text consumption and range 0–100. Accept '0', reject empty text, whitespace, trailing junk, and overflow. Do not return a view into local storage.",
                    cases: [
                        {input: "'0'", expected: "0 (engaged optional)"},
                        {input: "'100'", expected: "100"},
                        {input: "'75oops'", expected: "invalid"},
                        {input: "'101' or empty string", expected: "invalid"}
                    ],
                    hint: "Check empty input before pointer arithmetic; test result.ec and result.ptr, then the domain range. The optional owns an int, not a borrowed string.",
                    solution: `#include <charconv>
#include <iostream>
#include <optional>
#include <string_view>
#include <system_error>

std::optional<int> parse_score(std::string_view text) {
    if (text.empty()) return std::nullopt;
    int score{0};
    const char* end = text.data() + text.size();
    auto result = std::from_chars(text.data(), end, score);
    if (result.ec != std::errc{} || result.ptr != end || score < 0 || score > 100)
        return std::nullopt;
    return score;
}
int main() {
    for (std::string_view text : {"0", "100", "75oops", "101", ""}) {
        if (auto score = parse_score(text)) std::cout << *score << std::endl;
        else std::cout << "invalid" << std::endl;
    }
}
// The caller cannot confuse absent with zero. If detailed error reasons become a requirement, return variant<Score, ParseError> instead of overloading nullopt with hidden meaning.`
                },
                interview: {question: "Should return std::move(localVector) be used to make a factory faster?", answer: "Usually not. Returning the named local directly allows NRVO; if elision is not performed, eligible values can be moved. An explicit std::move can inhibit NRVO. Measure copies and performance rather than applying move everywhere."},
                readiness: ["I can distinguish absent, zero, malformed text, overflow, and range failure.", "I can explain the lifetime of string_view, the ownership of optional<int>, and why std::move is not itself a transfer."]
            },
            "Concurrency and Performance Engineering in C++": {
                prerequisites: "C++ Modules 1–4, RAII, lambdas, STL maps, and object lifetimes. Know that a C++ data race is undefined behavior, not just a sometimes-wrong counter.",
                setup: "Use C++20 with thread support. MSVC: cl /std:c++20 /EHsc /W4 main.cpp /Fe:logs.exe; GCC: g++ -std=c++20 -Wall -Wextra -pthread main.cpp -o logs.exe. Run .\\logs.exe in PowerShell. The workload is in memory; timings are not asserted.",
                workedExample: {
                    problem: "Aggregate error codes from two workers without concurrent map writes.",
                    steps: [
                        "Partition [E1,E2,E1] and [E2,E3]. Give each worker its own map, with lifetimes enclosing both threads. Each map is modified by only one worker.",
                        "Join both workers before the caller reads either result. Then merge by code on the caller thread.",
                        "Use std::map for sorted summary order. Totals become E1=2, E2=2, E3=1 regardless of worker scheduling. Sum of counts must equal five input events."
                    ],
                    expected: "E1=2 E2=2 E3=1 on each successful run. Two empty partitions print (none)."
                },
                practice: {
                    task: "Implement the partition-local aggregator below with joined lifetimes and sorted output. Acceptance: no shared writes; total counts equal input size; deterministic empty behavior. Compare serial and two-worker versions on the same generated input in an optimized build, report repeated measurements and checksum, and do not assume parallel is faster.",
                    cases: [
                        {input: "[E1,E2,E1] and [E2,E3]", expected: "E1=2 E2=2 E3=1; total=5"},
                        {input: "[] and []", expected: "(none); total=0"},
                        {input: "[E1] and [E1]", expected: "E1=2"}
                    ],
                    hint: "A separate map per worker removes the need for a map mutex. Join before merging; do not merely sleep and hope workers have finished. A real worker must capture/propagate exceptions rather than letting them escape a thread.",
                    solution: `#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

using Counts = std::map<std::string, int>;
void scan(const std::vector<std::string>& codes, Counts& out) {
    for (const auto& code : codes) ++out[code];
}
void aggregate(std::vector<std::string> a, std::vector<std::string> b) {
    Counts left, right;
    {
        std::jthread first([&] { scan(a, left); });
        std::jthread second([&] { scan(b, right); });
    }
    for (const auto& [code, count] : right) left[code] += count;
    if (left.empty()) std::cout << "(none)";
    bool separator{false};
    for (const auto& [code, count] : left) {
        if (separator) std::cout << ' ';
        std::cout << code << '=' << count;
        separator = true;
    }
    std::cout << std::endl;
}
int main() {
    aggregate({"E1", "E2", "E1"}, {"E2", "E3"});
    aggregate({}, {});
    aggregate({"E1"}, {"E1"});
}
// The jthread destructors join at the inner scope boundary before merging. Each worker owns its write target; only immutable input and caller-owned lifetimes are shared. This small example assumes successful allocation; production exception propagation needs explicit handling.`
                },
                interview: {question: "Would replacing every counter with an atomic make a whole report consistent?", answer: "No. Atomic increments can protect individual counters, not a snapshot or invariant spanning multiple values. Partition-local results followed by joining can give a consistent completed report with less contention. Benchmark overhead, allocation, and locality before changing synchronization."},
                readiness: ["I can prove which thread writes each map and where visibility is established before reading.", "I can compare serial/parallel checksums and report timings without claiming a guaranteed speedup."]
            },
            "Production C++ Capstone and Interview Readiness": {
                prerequisites: "C++ Modules 1–5, safe ownership, strict parsing, deterministic reporting, failure handling, and profiling basics. Choose the Log Analytics CLI option for this concrete scope.",
                setup: "Use a C++20 compiler with warnings and an existing build system or a documented single-command build. Start dependency-free with standard streams, from_chars, map, and a small assertion/test executable; tests must run with assertions enabled or use explicit failure checks. Work only on dedicated sample files. Add sanitizers only if the chosen toolchain supports them.",
                workedExample: {
                    problem: "Produce an exact log summary from a documented input format and preserve a prior report on validation failure.",
                    steps: [
                        "Define each nonempty row as CODE,COUNT, with no quoted fields: CODE is 1–16 uppercase ASCII letters/digits starting with a letter; COUNT is a full decimal nonnegative integer fitting uint64_t. Reject extra commas and detect total overflow.",
                        "For E1,2 then E2,1 then E1,3, parse into value records and merge into sorted totals E1=5 E2=1. An invalid row is reported with a one-based line number; this strict-mode run exits 2 and does not publish a new report.",
                        "Only after all rows validate, write a sibling staging report, check write/close success, then use the platform's documented replacement strategy. A failed publication must not be described as a successful run."
                    ],
                    expected: "Valid sample: E1=5 E2=1, exit 0. Append bad,row: reject line 4, exit 2, previous report unchanged. A missing input file exits 1."
                },
                practice: {
                    task: "Deliver logs --input sample.csv --output report.txt in strict mode. Acceptance: exact grammar and overflow checks; sorted deterministic totals; line-numbered rejects; missing/read/write failure handling; preserve old output on parse failure; >=12 automated parsing/aggregation/I/O tests; ownership notes; reproducible benchmark with checksum and peak-input size; README/demo and documented platform limits for report replacement. Optional parallel mode must match serial output byte-for-byte.",
                    cases: [
                        {input: "E1,2; E2,1; E1,3 on separate lines", expected: "report lines E1=5 and E2=1; exit 0"},
                        {input: "E1,2 then bad,row; prior report contains OLD", expected: "line 2 rejected; exit 2; report still OLD"},
                        {input: "E1,18446744073709551615 then E1,1", expected: "overflow rejection on line 2; no published report"}
                    ],
                    hint: "Check incoming > max - current before unsigned addition. Validate and aggregate before opening/truncating the final path. std::filesystem::rename replacement behavior must be verified on the target platform; do not promise cross-platform atomic replacement from its name alone.",
                    solution: "Implementation route: separate parse_row(string_view)->variant<Record,ParseError>, aggregate(records), and file orchestration. Enforce exactly one comma; validate ASCII code characters explicitly; parse unsigned COUNT with from_chars and require full consumption plus ec==errc{}. Before updating a total require count <= numeric_limits<uint64_t>::max()-total. Keep line numbers in errors and stop publication on any reject. For valid input, write the sorted report to a unique sibling staging file using an owning ofstream; explicitly flush/close and inspect failure state, then publish with a tested platform-specific strategy. Never delete the old report before a replacement guaranteed by that strategy. On unsupported replacement, return a clear I/O failure and retain the old file rather than silently weakening guarantees. Test missing source, malformed row, maximum count, overflow, empty input, output-open failure and cleanup. Benchmark serial first in an optimized build, consume its checksum, repeat fixed inputs, and document hardware/compiler/flags; compare optional parallel mode against identical golden output."
                },
                interview: {question: "What would you say if sanitizers and tests pass?", answer: "They provide evidence for exercised paths, not a proof of memory safety or production readiness. State the tested compiler/platform, sanitizer availability, input limits, untested failure modes, and replacement/durability assumptions. A benchmark only supports claims about its recorded workload."},
                readiness: ["I can demonstrate normal, malformed, overflow and I/O failure cases without damaging a prior report.", "I can explain every ownership boundary and present reproducible correctness/performance evidence with explicit limits."]
            }
        }
    };

    const expectedCounts = {javaProgramming: 6, pythonProgramming: 7, csharpProgramming: 6, cppProgramming: 6};
    const quizAnswers = {
        javaProgramming: ["B C B B B", "B B A B C", "C B B B C", "B B B B B", "B B B B B", null],
        pythonProgramming: ["B C B B B", "B B B B B", "B B B C B", "B B B B A", "B B B B B", null, null],
        csharpProgramming: ["B B B B B", "B B B B B", "B B B B B", "B B B B B", "B B B B B", null],
        cppProgramming: ["B B B B B", "B B B B B", "B B B B B", "B B B B B", "B B B B B", null]
    };

    for (const [track, count] of Object.entries(expectedCounts)) {
        const modules = courseData[track];
        const trackGuides = guides[track];
        if (!Array.isArray(modules) || modules.length !== count ||
            Object.keys(trackGuides).length !== count) {
            throw new Error(`Programming curriculum count mismatch: ${track}`);
        }
        const seen = new Set();
        modules.forEach((module, index) => {
            if (module.number !== `Module ${index + 1}` || seen.has(module.title) ||
                !Object.prototype.hasOwnProperty.call(trackGuides, module.title)) {
                throw new Error(`Missing or mismatched learning guide: ${track} / ${module.number} / ${module.title}`);
            }
            seen.add(module.title);
            module.learningGuide = trackGuides[module.title];

            if (index === 0) {
                const historyIndex = module.detailedContent.findIndex(lesson => lesson.title.startsWith("Optional:"));
                const projectIndex = module.detailedContent.findIndex(lesson => lesson.title.startsWith("Mini Project:"));
                if (historyIndex < 0 || projectIndex <= historyIndex) {
                    throw new Error(`Missing first-program sequence anchors: ${track}`);
                }
                const [history] = module.detailedContent.splice(historyIndex, 1);
                module.detailedContent.splice(projectIndex - 1, 0, history);
            }

            const quiz = module.detailedContent.find(lesson => lesson.title === "Module Quiz");
            const answers = quizAnswers[track][index];
            if (Boolean(quiz) !== Boolean(answers)) {
                throw new Error(`Missing quiz answer key: ${track} / ${module.number}`);
            }
            if (quiz) {
                quiz.content += `\nAnswer key (questions 1–5): ${answers.split(" ").join(", ")}. Review the corresponding lesson and explain why the other choices do not fit.`;
            }
        });
    }
})();
