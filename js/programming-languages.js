// ============================================================
// Programming Languages track content.
// Loaded only on the Courses page (after script.js). It extends the
// existing global `courseData` object with language-specific tracks.
// ============================================================

/* global courseData */

courseData.javaProgramming = [
    {
        number: "Module 1",
        title: "Java Fundamentals",
        description: "Start from zero and build strong Java foundations with syntax, control flow, arrays, strings, and practical coding habits.",
        duration: "60 min",
        lessons: "12 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Java Introduction", "JDK vs JRE vs JVM", "Installing Java", "Hello World", "Variables", "Data Types", "Operators", "User Input", "Type Casting", "Control Flow", "Loops", "Methods", "Arrays", "Strings", "Wrapper Classes", "Math Utilities", "Mini Assessment"],
        detailedDescription: "Beginner-first module that establishes Java runtime understanding and coding basics required for all later modules.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 1
Module Name: Java Fundamentals
Difficulty: Beginner
Estimated Reading Time: 60 min
Estimated Completion Time: 5-6 hours
Prerequisites: None
Learning Objectives:
• understand Java runtime model and tooling
• write correct Java syntax with confidence
• use core language constructs in small programs
Skills Gained:
• setup and execute Java programs
• use variables, data types, and operators
• apply control flow, loops, methods, arrays, and strings`
            },
            {
                title: "Lesson 1: Java Introduction and Runtime Setup",
                content: `Learning Objective: Understand Java ecosystem and run first program.
Estimated Reading Time: 8 min
Difficulty: Beginner
Theory Content: Java source is compiled to bytecode and executed by JVM. JDK includes compiler tools, JRE includes runtime libraries.
Real World Analogy: Recipe (source), prepped kit (bytecode), chef station (JVM).
Visual Diagram (Markdown): Write .java -> javac -> .class -> java -> output
Expected Output: Hello, Java
Common Mistakes: wrong class/file names, missing main method signature.
Best Practices: install latest LTS JDK and verify PATH.
Mini Exercise: create and run HelloWorld.java.
Key Takeaways: JDK builds, JVM runs, Java is platform independent.`,
                code: `public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, Java");
    }
}`
            },
            {
                title: "Lesson 2: Variables, Data Types, and Operators",
                content: `Learning Objective: Store and manipulate data safely.
Estimated Reading Time: 10 min
Difficulty: Beginner
Theory Content: Primitive types define value ranges and memory behavior; operators perform arithmetic and logic.
Real World Analogy: Labeled containers holding specific material types.
Visual Diagram (Markdown): int/double/boolean/String -> operations -> result
Expected Output: computed totals and boolean checks.
Common Mistakes: implicit narrowing, integer division confusion.
Best Practices: choose the smallest safe numeric type and keep expressions readable.
Mini Exercise: compute simple bill with tax and discount.
Key Takeaways: type choice affects correctness and output.`,
                code: `int qty = 3;
double price = 199.99;
double total = qty * price;
boolean highValue = total > 500;
System.out.println(total);
System.out.println(highValue);`
            },
            {
                title: "Lesson 3: User Input and Type Casting",
                content: `Learning Objective: Read external input and cast values correctly.
Estimated Reading Time: 8 min
Difficulty: Beginner
Theory Content: Scanner reads typed input, explicit casting controls conversion precision.
Real World Analogy: translating numbers between currencies with rounding rules.
Visual Diagram (Markdown): input text -> parse -> cast -> compute
Expected Output: personalized and computed values.
Common Mistakes: leaving newline in Scanner buffer.
Best Practices: validate input before casting.
Mini Exercise: read age and print age in months.
Key Takeaways: conversion and validation are core runtime skills.`,
                code: `import java.util.Scanner;

Scanner sc = new Scanner(System.in);
System.out.print("Enter score: ");
double score = sc.nextDouble();
int rounded = (int) score;
System.out.println("Rounded score: " + rounded);`
            },
            {
                title: "Lesson 4: Control Flow and Loops",
                content: `Learning Objective: Control decision paths and repetition.
Estimated Reading Time: 10 min
Difficulty: Beginner
Theory Content: if-else routes logic; for/while repeat work until condition changes.
Real World Analogy: traffic signals deciding which lane moves and for how long.
Visual Diagram (Markdown): condition -> branch A/B -> loop -> stop
Expected Output: branch-specific messages and loop sequences.
Common Mistakes: infinite loops and wrong condition order.
Best Practices: keep loop logic minimal and predictable.
Mini Exercise: print even numbers from 1 to 20.
Key Takeaways: flow control drives program behavior.`,
                code: `for (int i = 1; i <= 10; i++) {
    if (i % 2 == 0) {
        System.out.println(i + " is even");
    }
}`
            },
            {
                title: "Lesson 5: Methods, Arrays, and Strings",
                content: `Learning Objective: Write reusable logic and process collections/text.
Estimated Reading Time: 12 min
Difficulty: Beginner
Theory Content: methods encapsulate behavior; arrays store indexed values; String APIs handle text tasks.
Real World Analogy: reusable machine tool applied to multiple items on a conveyor.
Visual Diagram (Markdown): method(input) -> process array -> string output
Expected Output: transformed values and formatted text.
Common Mistakes: off-by-one index errors.
Best Practices: create focused methods with clear names.
Mini Exercise: write method that finds max value in an int array.
Key Takeaways: modularity and data traversal form coding foundation.`,
                code: `static String buildMessage(String name, int[] marks) {
    int sum = 0;
    for (int m : marks) sum += m;
    return name + " average = " + (sum / marks.length);
}

System.out.println(buildMessage("Ravi", new int[]{80, 90, 70}));`
            },
            {
                title: "Lesson 6: Wrapper Classes and Math Utilities",
                content: `Learning Objective: Use utility classes for robust numeric logic.
Estimated Reading Time: 7 min
Difficulty: Beginner
Theory Content: wrappers bridge primitives and objects; Math offers common numeric operations.
Real World Analogy: toolkit add-ons for precise measurement and conversion.
Visual Diagram (Markdown): primitive <-> wrapper -> Math operations
Expected Output: parsed and rounded values.
Common Mistakes: NullPointerException from unboxing null wrappers.
Best Practices: parse safely and validate before conversion.
Mini Exercise: parse string prices and round totals.
Key Takeaways: wrappers and Math utilities appear in real business logic.`,
                code: `String amount = "199.75";
double value = Double.parseDouble(amount);
long rounded = Math.round(value);
System.out.println(rounded);`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
• build StudentResultConsole to read student name and 3 marks
• calculate average and grade using reusable grading methods

Success Check:
• output shows correct grade boundaries for sample inputs
• final result card is clean, readable, and consistently formatted`
            },
            {
                title: "Module Quiz",
                content: `1) JVM executes: A) source B) bytecode C) docs D) jar metadata
2) Which input class is common in console apps? A) Scanner B) StringBuilder C) FileWriter D) Thread
3) int division 5/2 gives: A) 2.5 B) 2 C) 3 D) error
4) Array index starts at: A) 1 B) 0 C) -1 D) depends
5) Math.round returns closest: A) int B) long C) double D) float`
            },
            {
                title: "Interview Preparation",
                content: `Common interview prompts:
• Explain JDK vs JRE vs JVM.
• Why is Java platform independent?
• Difference between primitive and wrapper types.
• Predict output questions using loops and conditions.
Follow-up: optimize readability and avoid edge-case bugs.`
            },
            {
                title: "Module Summary",
                content: `You can now:
• set up and run Java programs
• write core syntax with confidence
• solve small logic problems using methods, arrays, and strings
• prepare for OOP transition`
            },
            {
                title: "Next Module Bridge",
                content: `Next module introduces Object-Oriented Programming where you model real entities using classes, objects, and reusable contracts.`
            }
        ]
    },
    {
        number: "Module 2",
        title: "Object-Oriented Programming",
        description: "Learn complete Java OOP from class design to object contracts and build a Library Management mini project.",
        duration: "65 min",
        lessons: "12 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Classes", "Objects", "Constructors", "Encapsulation", "Inheritance", "Polymorphism", "Abstraction", "Interfaces", "Packages", "Access Modifiers", "Static", "Final", "Object Class", "equals()", "hashCode()", "toString()", "Mini Project: Library Management System"],
        detailedDescription: "Industry-style OOP progression focused on maintainability, extensibility, and interview-ready reasoning.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 2
Module Name: Object-Oriented Programming
Difficulty: Beginner to Intermediate
Estimated Reading Time: 65 min
Estimated Completion Time: 6-7 hours
Prerequisites: Module 1
Learning Objectives:
• model software entities with classes and objects
• apply encapsulation, inheritance, polymorphism, abstraction
• implement robust object contracts with equals/hashCode/toString
Skills Gained:
• OOP design thinking
• reusable class architecture
• object identity and consistency handling`
            },
            {
                title: "Lesson 1: Classes, Objects, Constructors",
                content: `Learning Objective: Build valid object state through constructors.
Estimated Reading Time: 9 min
Difficulty: Beginner
Theory Content: class defines blueprint, object is instance, constructor initializes mandatory state.
Real World Analogy: apartment blueprint vs actual apartment unit.
Visual Diagram (Markdown): Class -> new -> Object(state)
Expected Output: object fields initialized predictably.
Common Mistakes: forgetting constructor arguments or shadowing fields.
Best Practices: validate constructor inputs.
Mini Exercise: create Book class with title and isbn.
Key Takeaways: object integrity starts at construction.`,
                code: `class Book {
    private final String title;
    Book(String title) { this.title = title; }
    String getTitle() { return title; }
}`
            },
            {
                title: "Lesson 2: Encapsulation and Access Modifiers",
                content: `Learning Objective: Protect state and expose behavior safely.
Estimated Reading Time: 8 min
Difficulty: Beginner
Theory Content: private/protected/public control visibility and maintain invariants.
Real World Analogy: locker with controlled key access.
Visual Diagram (Markdown): private field -> public method -> validated update
Expected Output: only valid updates allowed.
Common Mistakes: making all fields public.
Best Practices: prefer private fields with intent-revealing methods.
Mini Exercise: add deposit() validation in Account class.
Key Takeaways: encapsulation reduces fragile code.`
            },
            {
                title: "Lesson 3: Inheritance and Polymorphism",
                content: `Learning Objective: Reuse behavior correctly via subtype contracts.
Estimated Reading Time: 10 min
Difficulty: Intermediate
Theory Content: subclass extends base behavior; polymorphism enables runtime method dispatch.
Real World Analogy: general vehicle rules specialized by car and bike.
Visual Diagram (Markdown): BaseType ref -> Child override at runtime
Expected Output: overridden method is invoked.
Common Mistakes: deep inheritance trees without need.
Best Practices: prefer composition if inheritance is weak.
Mini Exercise: create Notification base and Email/SMS subtypes.
Key Takeaways: polymorphism enables extensible systems.`,
                code: `class Animal { void speak() { System.out.println("..."); } }
class Dog extends Animal { @Override void speak() { System.out.println("Bark"); } }
Animal a = new Dog();
a.speak();`
            },
            {
                title: "Lesson 4: Abstraction and Interfaces",
                content: `Learning Objective: Separate contract from implementation.
Estimated Reading Time: 9 min
Difficulty: Intermediate
Theory Content: abstraction hides complexity, interfaces define behavior contracts.
Real World Analogy: power socket standard with different appliance designs.
Visual Diagram (Markdown): Interface -> multiple implementations
Expected Output: interchangeable implementations.
Common Mistakes: putting heavy state logic in interfaces.
Best Practices: design stable interfaces around behavior.
Mini Exercise: create Payment interface with CardPayment and UpiPayment.
Key Takeaways: interfaces improve testability and flexibility.`
            },
            {
                title: "Lesson 5: Packages, Static, and Final",
                content: `Learning Objective: Organize codebase and use class-level semantics.
Estimated Reading Time: 8 min
Difficulty: Intermediate
Theory Content: packages provide namespace and modularity; static is class-level; final locks variable/method/class semantics.
Real World Analogy: city -> district -> building addressing.
Visual Diagram (Markdown): package -> class -> static member
Expected Output: stable constants and utility access.
Common Mistakes: overusing static mutable state.
Best Practices: use static for stateless helpers/constants.
Mini Exercise: move utility methods into dedicated package.
Key Takeaways: structure and intent improve maintainability.`
            },
            {
                title: "Lesson 6: Object Class, equals(), hashCode(), toString()",
                content: `Learning Objective: Implement object contracts correctly.
Estimated Reading Time: 11 min
Difficulty: Intermediate
Theory Content: equals/hashCode consistency is mandatory for map/set behavior; toString aids debugging.
Real World Analogy: identity card rules in a registry system.
Visual Diagram (Markdown): equals true => same hashCode
Expected Output: duplicate detection works in HashSet.
Common Mistakes: overriding equals without hashCode.
Best Practices: use immutable key fields in contract methods.
Mini Exercise: create Member class and test HashSet behavior.
Key Takeaways: object contracts are interview-critical.`,
                code: `class Member {
    private final String id;
    Member(String id) { this.id = id; }
    @Override public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Member)) return false;
        return id.equals(((Member) o).id);
    }
    @Override public int hashCode() { return id.hashCode(); }
    @Override public String toString() { return "Member(" + id + ")"; }
}`
            },
            {
                title: "Mini Project: Library Management System",
                content: `Project Goal: build a clean OOP-based console app that manages books, members, and loans reliably.

Domain Modeling:
• define entities: Book, Member, Loan
• design class responsibilities with encapsulated state
• enforce identity rules using equals/hashCode where needed

Core Workflow:
• add and search books
• register members
• issue and return books with validation checks

Architecture Expectations:
• package separation: model, service, app
• no business logic in CLI/UI layer
• readable method names and cohesive classes

Evaluation Signals:
• clear OOP design and class interaction
• correct issue/return constraints
• maintainable package structure for future features`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
• add overdue fee calculation and member borrowing limit rules
• enforce these rules in issue/return workflows without API breakage

Success Check:
• invalid issue requests are blocked with clear reason
• existing add/search/issue/return flows still work as expected`
            },
            {
                title: "Module Quiz",
                content: `1) Polymorphism means: A) one class B) one interface only C) one reference, many forms D) no inheritance
2) Best encapsulation choice: A) public fields B) private fields + methods C) static globals D) package-private everything
3) equals/hashCode relation: A) unrelated B) must be consistent C) hashCode optional always D) toString decides
4) final class can be: A) extended B) not extended C) abstract only D) interface
5) static member belongs to: A) object instance B) class C) JVM thread D) package`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus areas:
• explain four OOP pillars with examples
• interface vs abstract class trade-offs
• equals/hashCode contract scenarios
• design a small domain model live`
            },
            {
                title: "Module Summary",
                content: `You can now design Java classes with production-safe contracts and package-level organization.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module expands into collections, generics, and exception handling for scalable data-heavy logic.`
            }
        ]
    },
    {
        number: "Module 3",
        title: "Collections, Generics & Exception Handling",
        description: "Master Java data structures, type-safe abstractions, and robust failure handling with a Student Record mini project.",
        duration: "70 min",
        lessons: "12 lessons",
        isNew: false,
        isLocked: false,
        topics: ["List", "Set", "Queue", "Map", "Generic Classes", "Generic Methods", "Wildcards", "try", "catch", "finally", "throw", "throws", "Custom Exceptions", "Mini Project: Student Record Manager"],
        detailedDescription: "Structured module for handling collections at scale and writing resilient Java code.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 3
Module Name: Collections, Generics & Exception Handling
Difficulty: Intermediate
Estimated Reading Time: 70 min
Estimated Completion Time: 7-8 hours
Prerequisites: Modules 1-2
Learning Objectives:
• choose right collection for each access pattern
• design generic reusable components
• handle and propagate exceptions correctly
Skills Gained:
• collection performance trade-offs
• compile-time type safety
• production-grade error handling`
            },
            {
                title: "Lesson 1: Collections Framework (List, Set, Queue, Map)",
                content: `Learning Objective: Choose data structures intentionally.
Estimated Reading Time: 12 min
Difficulty: Intermediate
Theory Content: List preserves order, Set enforces uniqueness, Queue supports FIFO workflows, Map handles key lookup.
Real World Analogy: line queue, unique guest list, address book.
Visual Diagram (Markdown): Input -> choose structure -> operations
Expected Output: predictable lookup/update behavior.
Common Mistakes: using wrong structure then compensating with extra code.
Best Practices: optimize by operation frequency, not habit.
Mini Exercise: refactor one problem using two different structures.
Key Takeaways: structure choice drives performance.`
            },
            {
                title: "Lesson 2: Generic Classes, Methods, and Wildcards",
                content: `Learning Objective: Write reusable type-safe APIs.
Estimated Reading Time: 10 min
Difficulty: Intermediate
Theory Content: Generics prevent unsafe casts; wildcards define variance flexibility.
Real World Analogy: container labels that restrict what may be stored.
Visual Diagram (Markdown): Box<T> / method<T> / List<? extends Number>
Expected Output: compile-time safety without casting noise.
Common Mistakes: raw types and wildcard misuse.
Best Practices: prefer bounded generics for expressive APIs.
Mini Exercise: build generic Pair<K,V> class.
Key Takeaways: generics improve correctness and readability.`,
                code: `class Box<T> {
    private T value;
    void set(T value) { this.value = value; }
    T get() { return value; }
}`
            },
            {
                title: "Lesson 3: try, catch, finally",
                content: `Learning Objective: Recover safely from known failure cases.
Estimated Reading Time: 8 min
Difficulty: Intermediate
Theory Content: try encloses risk, catch handles known failures, finally ensures cleanup.
Real World Analogy: safety protocol after machine fault.
Visual Diagram (Markdown): try -> catch? -> finally always
Expected Output: graceful failure handling.
Common Mistakes: swallowing exceptions silently.
Best Practices: catch specific exceptions first.
Mini Exercise: handle division and parsing errors in one workflow.
Key Takeaways: predictable failure handling is a quality signal.`
            },
            {
                title: "Lesson 4: throw, throws, and Custom Exceptions",
                content: `Learning Objective: Model business errors explicitly.
Estimated Reading Time: 8 min
Difficulty: Intermediate
Theory Content: throw emits exception now; throws declares contract; custom exceptions convey domain context.
Real World Analogy: structured incident ticket with clear category.
Visual Diagram (Markdown): validation fail -> throw DomainException
Expected Output: meaningful error messages.
Common Mistakes: generic RuntimeException with no context.
Best Practices: custom exception names should explain business rule breach.
Mini Exercise: create InvalidScoreException.
Key Takeaways: domain exceptions improve debugging and API clarity.`,
                code: `class InvalidScoreException extends RuntimeException {
    InvalidScoreException(String msg) { super(msg); }
}`
            },
            {
                title: "Lesson 5: Integrated Data Pipeline with Collections + Exceptions",
                content: `Learning Objective: Combine collections and exception logic in one cohesive flow.
Estimated Reading Time: 10 min
Difficulty: Intermediate
Theory Content: parse, validate, store, and query records through robust steps.
Real World Analogy: warehouse intake with quality checks and categorized bins.
Visual Diagram (Markdown): read -> validate -> map/list -> report
Expected Output: valid records processed, invalid records reported.
Common Mistakes: partial updates without rollback strategy.
Best Practices: separate parse, validate, and store responsibilities.
Mini Exercise: collect invalid rows separately.
Key Takeaways: composable pipelines are production-friendly.`
            },
            {
                title: "Mini Project: Student Record Manager",
                content: `Project Goal: build a resilient student performance manager with structured data and explicit error handling.

Data Design:
• maintain score history using Map<String, List<Integer>>
• keep add/update/query flows predictable
• prevent invalid or partial updates

Reliability Requirements:
• create custom exceptions for invalid marks, missing students, and malformed input
• separate validation from storage logic
• preserve consistent state after failures

Analytics Output:
• generate class summary (average, topper, low performer)
• produce per-student progress snapshot
• report invalid rows with clear reason codes

Evaluation Signals:
• right collection choices for operations
• meaningful exception messages
• accurate report generation across edge cases`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
• add CSV import for student marks and metadata
• handle malformed rows without stopping the full import run

Success Check:
• rejected rows are counted with explicit error reasons
• valid rows are imported and reflected in summary reports`
            },
            {
                title: "Module Quiz",
                content: `1) Which structure guarantees unique elements? A) List B) Set C) Queue D) Map
2) Generics primarily improve: A) syntax color B) type safety C) disk size D) startup speed
3) finally block executes: A) only on success B) only on error C) always (except JVM halt) D) never
4) throws keyword is used to: A) throw now B) declare propagation C) catch error D) log message
5) Best for key-value lookup: A) Queue B) Map C) Set D) Array`
            },
            {
                title: "Interview Preparation",
                content: `Interview themes:
• compare List/Set/Map by complexity and use-case
• explain PECS (producer extends, consumer super)
• checked vs unchecked exception strategy
• design record manager with validation and error handling`
            },
            {
                title: "Module Summary",
                content: `You can now build data-centric Java logic with correct structure choices, generic APIs, and explicit failure contracts.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module introduces modern Java features that make data processing concise and expressive.`
            }
        ]
    },
    {
        number: "Module 4",
        title: "Modern Java Programming",
        description: "Adopt modern Java with functional style, stream processing, Optional, records, and Java 21-aligned features.",
        duration: "65 min",
        lessons: "12 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Functional Interfaces", "Lambda Expressions", "Method References", "Streams API", "Optional", "Date & Time API", "Records", "Sealed Classes", "Pattern Matching (Java 21 if applicable)", "Mini Project: Employee Analytics using Streams"],
        detailedDescription: "Intermediate module focused on concise, expressive, and modern Java coding patterns.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 4
Module Name: Modern Java Programming
Difficulty: Intermediate
Estimated Reading Time: 65 min
Estimated Completion Time: 6-7 hours
Prerequisites: Modules 1-3
Learning Objectives:
• write functional-style Java code
• process collections with Streams API
• model optionality and modern data carriers
Skills Gained:
• lambda fluency
• stream pipeline design
• modern language feature adoption`
            },
            {
                title: "Lesson 1: Functional Interfaces, Lambdas, Method References",
                content: `Learning Objective: Pass behavior as data cleanly.
Estimated Reading Time: 12 min
Difficulty: Intermediate
Theory Content: functional interfaces support single abstract method; lambdas and method references provide concise implementations.
Real World Analogy: plug different tools into one machine port.
Visual Diagram (Markdown): Function contract -> lambda -> execute
Expected Output: transformed values via function pipeline.
Common Mistakes: verbose lambdas where method reference is clearer.
Best Practices: favor readability over compactness.
Mini Exercise: convert anonymous class to lambda.
Key Takeaways: behavior abstraction reduces boilerplate.`,
                code: `List<String> names = Arrays.asList("ravi", "asha");
names.stream().map(String::toUpperCase).forEach(System.out::println);`
            },
            {
                title: "Lesson 2: Streams API and Optional",
                content: `Learning Objective: Build readable transformation pipelines and null-safe flows.
Estimated Reading Time: 12 min
Difficulty: Intermediate
Theory Content: stream stages (filter/map/reduce/collect), Optional for absent values.
Real World Analogy: assembly line with quality gate and packaging.
Visual Diagram (Markdown): source -> filter -> map -> collect
Expected Output: selected and transformed dataset.
Common Mistakes: side effects inside stream operations.
Best Practices: keep pipelines short and composable.
Mini Exercise: filter top scores and map names.
Key Takeaways: modern data processing should be explicit and safe.`,
                code: `List<Integer> top = Arrays.asList(50, 90, 82, 97).stream()
    .filter(v -> v >= 85)
    .sorted()
    .collect(java.util.stream.Collectors.toList());
System.out.println(top);`
            },
            {
                title: "Lesson 3: Date & Time API",
                content: `Learning Objective: Use immutable date-time types correctly.
Estimated Reading Time: 8 min
Difficulty: Intermediate
Theory Content: LocalDate/LocalDateTime/Duration provide safer handling than legacy Date.
Real World Analogy: standardized calendar rules instead of handwritten date math.
Visual Diagram (Markdown): parse -> calculate -> format
Expected Output: accurate date operations.
Common Mistakes: mixing timezone assumptions.
Best Practices: store explicit timezone where needed.
Mini Exercise: compute days between two dates.
Key Takeaways: java.time is the default for modern systems.`
            },
            {
                title: "Lesson 4: Records, Sealed Classes, Pattern Matching",
                content: `Learning Objective: Model domain types with less boilerplate and safer branching.
Estimated Reading Time: 10 min
Difficulty: Intermediate
Theory Content: records provide immutable data carriers; sealed classes restrict hierarchies; pattern matching improves type checks.
Real World Analogy: approved role list with enforced access boundaries.
Visual Diagram (Markdown): sealed hierarchy -> exhaustive branching
Expected Output: concise immutable models and clear branch logic.
Common Mistakes: forcing records where mutable behavior is required.
Best Practices: use records for DTO/value objects.
Mini Exercise: create record EmployeeSummary.
Key Takeaways: modern Java improves clarity and correctness.`
            },
            {
                title: "Lesson 5: End-to-End Functional Refactor",
                content: `Learning Objective: Refactor imperative code to modern style responsibly.
Estimated Reading Time: 9 min
Difficulty: Intermediate
Theory Content: identify loop-heavy hotspots and convert incrementally to streams.
Real World Analogy: replacing manual assembly with semi-automated stations.
Visual Diagram (Markdown): old loop -> extraction -> stream pipeline
Expected Output: same behavior with cleaner implementation.
Common Mistakes: rewriting everything at once.
Best Practices: validate output parity before and after refactor.
Mini Exercise: refactor one report method.
Key Takeaways: modernization should preserve behavior.`
            },
            {
                title: "Mini Project: Employee Analytics using Streams",
                content: `Project Goal: build a modern analytics console using streams, Optional, and record-based output models.

Data Pipeline:
• group employees by department
• compute average salary, max salary, and top performer per group
• filter and rank insights with readable stream pipelines

Null-Safety and Correctness:
• use Optional for missing manager relationships
• avoid unsafe null checks and side effects inside streams
• keep transformations deterministic and testable

Reporting Layer:
• expose report rows as record-based DTOs
• generate concise terminal summary for stakeholders
• include one anomaly section (missing manager, outlier salary, or empty team)

Evaluation Signals:
• stream pipeline clarity over clever one-liners
• correct aggregate calculations
• strong explanation of map/filter/reduce decisions`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
• add monthly hiring trend analysis using java.time and stream grouping
• include trend output by department and overall totals

Success Check:
• trend calculations match raw data sample checks
• output remains readable with no stream side-effect bugs`
            },
            {
                title: "Module Quiz",
                content: `1) Functional interface has: A) many abstract methods B) one abstract method C) none D) only default methods
2) Optional helps avoid: A) syntax errors B) NullPointerException patterns C) compile warnings only D) memory leaks
3) Stream terminal op example: A) map B) filter C) collect D) sorted
4) Record is best for: A) mutable entities B) immutable data carriers C) threads D) interfaces only
5) Sealed classes primarily control: A) package imports B) subclassing scope C) JVM heap D) logging`
            },
            {
                title: "Interview Preparation",
                content: `Interview prompts:
• map vs flatMap
• reduce vs collect
• Optional anti-patterns
• when to use records/sealed classes
• convert legacy loop to stream with explanation`
            },
            {
                title: "Module Summary",
                content: `You can now write modern Java with functional constructs and concise domain modeling techniques.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module brings multithreading and production runtime concerns to make your Java systems scale safely.`
            }
        ]
    },
    {
        number: "Module 5",
        title: "Multithreading & Production Java",
        description: "Learn concurrency, runtime behavior, and production engineering basics for stable Java systems.",
        duration: "75 min",
        lessons: "12 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Threads", "Runnable", "Synchronization", "Executors", "CompletableFuture", "JVM Memory", "Garbage Collection", "File Handling", "Serialization", "Logging", "Configuration", "Performance Basics", "Mini Project: Multithreaded File Processor"],
        detailedDescription: "Advanced-intermediate module focused on reliability and performance under realistic workloads.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 5
Module Name: Multithreading & Production Java
Difficulty: Intermediate to Advanced
Estimated Reading Time: 75 min
Estimated Completion Time: 8-9 hours
Prerequisites: Modules 1-4
Learning Objectives:
• run concurrent tasks safely
• understand JVM memory and GC basics
• apply logging, configuration, and performance fundamentals
Skills Gained:
• thread-safe design
• async processing patterns
• production troubleshooting mindset`
            },
            {
                title: "Lesson 1: Threads, Runnable, Synchronization",
                content: `Learning Objective: coordinate shared state safely.
Estimated Reading Time: 12 min
Difficulty: Intermediate
Theory Content: thread lifecycle, Runnable task model, synchronized critical sections.
Real World Analogy: multiple clerks writing to one ledger.
Visual Diagram (Markdown): thread A/B -> shared resource -> synchronized lock
Expected Output: deterministic counter updates.
Common Mistakes: unsynchronized shared mutation.
Best Practices: keep synchronized blocks minimal.
Mini Exercise: thread-safe counter increment.
Key Takeaways: correctness before concurrency speed.`
            },
            {
                title: "Lesson 2: Executors and CompletableFuture",
                content: `Learning Objective: manage async tasks with scalable abstractions.
Estimated Reading Time: 12 min
Difficulty: Intermediate
Theory Content: ExecutorService controls pools; CompletableFuture composes async pipelines.
Real World Analogy: dispatch center assigning jobs to workers.
Visual Diagram (Markdown): submit -> future -> combine -> result
Expected Output: composed async results.
Common Mistakes: blocking immediately after async submit.
Best Practices: use timeouts and explicit error stages.
Mini Exercise: combine two async API simulations.
Key Takeaways: managed concurrency beats manual threads.`,
                code: `java.util.concurrent.CompletableFuture<Integer> a =
    java.util.concurrent.CompletableFuture.supplyAsync(() -> 40);
java.util.concurrent.CompletableFuture<Integer> b =
    java.util.concurrent.CompletableFuture.supplyAsync(() -> 2);
System.out.println(a.thenCombine(b, Integer::sum).join());`
            },
            {
                title: "Lesson 3: JVM Memory and Garbage Collection",
                content: `Learning Objective: reason about memory usage and GC behavior.
Estimated Reading Time: 10 min
Difficulty: Intermediate
Theory Content: heap vs stack, object lifetime, young/old generation concepts.
Real World Analogy: short-stay and long-stay storage rooms.
Visual Diagram (Markdown): allocation -> survivor -> old gen -> GC cycle
Expected Output: reduced memory pressure with better object lifecycle.
Common Mistakes: retaining references accidentally.
Best Practices: avoid unnecessary object churn in hot loops.
Mini Exercise: identify leak-like retention pattern.
Key Takeaways: memory literacy improves stability.`
            },
            {
                title: "Lesson 4: File Handling and Serialization",
                content: `Learning Objective: persist and restore data safely.
Estimated Reading Time: 10 min
Difficulty: Intermediate
Theory Content: stream/file APIs for text and object persistence; serialization caveats.
Real World Analogy: filing cabinet with strict format rules.
Visual Diagram (Markdown): object -> stream -> file -> stream -> object
Expected Output: stored and reloaded records.
Common Mistakes: missing version compatibility handling.
Best Practices: prefer explicit data formats for long-lived storage.
Mini Exercise: save and reload list of records.
Key Takeaways: persistence requires format discipline.`
            },
            {
                title: "Lesson 5: Logging, Configuration, Performance Basics",
                content: `Learning Objective: make applications observable and tunable.
Estimated Reading Time: 10 min
Difficulty: Intermediate
Theory Content: structured logs, environment-driven config, basic profiling mindset.
Real World Analogy: cockpit dashboard for flight monitoring.
Visual Diagram (Markdown): app events -> logger -> file/console -> analysis
Expected Output: actionable logs and configurable behavior.
Common Mistakes: logging secrets or noisy debug output in production.
Best Practices: log context-rich events with levels.
Mini Exercise: externalize one config value and log startup config.
Key Takeaways: observability is non-negotiable in production.`
            },
            {
                title: "Mini Project: Multithreaded File Processor",
                content: `Project Goal: build a production-style file processor that handles batch input safely and fast.

Core Workflow:
• read multiple files concurrently using ExecutorService
• validate each line against expected format rules
• collect valid records and reject invalid entries

Concurrency Requirements:
• isolate per-file processing to avoid shared-state bugs
• capture failures without stopping full batch
• keep final counters thread-safe

Output Requirements:
• aggregated summary (total files, processed lines, valid lines, invalid lines)
• error log with file name, line number, and reason
• end-of-run report ready for operations review

Evaluation Signals:
• handles bad files gracefully
• produces deterministic summary across runs
• clear logs and readable report for debugging`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
• add retry strategy for transient file read failures
• capture processing time per file and include it in summary report

Success Check:
• transient failures recover within retry limit
• per-file timing and aggregate timing are logged accurately`
            },
            {
                title: "Module Quiz",
                content: `1) Runnable represents: A) thread pool B) task without return C) JVM process D) exception handler
2) CompletableFuture is best for: A) static config B) async composition C) serialization only D) package imports
3) GC primarily manages: A) CPU threads B) memory reclamation C) classpath D) network sockets
4) Production logging should avoid: A) context B) sensitive data exposure C) levels D) timestamps
5) Synchronization prevents: A) compile errors B) race conditions C) package conflicts D) class loading`
            },
            {
                title: "Interview Preparation",
                content: `Interview areas:
• explain race condition with practical example
• Runnable vs Callable vs CompletableFuture
• heap/stack and GC basics
• diagnose slow program using logs and simple profiling`
            },
            {
                title: "Module Summary",
                content: `You can now build reliable concurrent Java workflows with production-oriented diagnostics and runtime awareness.`
            },
            {
                title: "Next Module Bridge",
                content: `Final module combines all learning in one complete real-world capstone and final assessment.`
            }
        ]
    },
    {
        number: "Module 6",
        title: "Real-World Java Capstone Project",
        description: "Build a complete console-based Java application with architecture, persistence, validation, and interview-ready documentation.",
        duration: "90 min",
        lessons: "12 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Build a complete console app", "Clean architecture", "OOP", "Collections", "Exception Handling", "File Storage", "Modular Code", "Documentation", "Suggested Projects: Banking", "Suggested Projects: Library", "Suggested Projects: Inventory", "Suggested Projects: Expense Tracker", "Suggested Projects: Student Management", "Final Assessment"],
        detailedDescription: "Capstone module to transition from learner to job-ready builder with complete project delivery and assessment.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 6
Module Name: Real-World Java Capstone Project
Difficulty: Advanced
Estimated Reading Time: 90 min
Estimated Completion Time: 10-12 hours
Prerequisites: Modules 1-5
Learning Objectives:
• architect and ship complete console project
• integrate OOP, collections, exception handling, and file storage
• present solution with documentation and interview narrative
Skills Gained:
• end-to-end implementation
• project decomposition and modular code
• job-ready communication of technical decisions`
            },
            {
                title: "Lesson 1: Requirements and Scope Definition",
                content: `Learning Objective: define realistic project boundaries.
Estimated Reading Time: 10 min
Difficulty: Advanced
Theory Content: identify core entities, actions, constraints, and success criteria.
Real World Analogy: writing architectural brief before constructing a building.
Visual Diagram (Markdown): requirements -> modules -> milestones
Expected Output: clear project scope document.
Common Mistakes: over-scoping first version.
Best Practices: prioritize MVP features first.
Mini Exercise: define MVP for one suggested project.
Key Takeaways: scope controls delivery quality.`
            },
            {
                title: "Lesson 2: Clean Architecture and Modular Code",
                content: `Learning Objective: separate concerns into stable layers.
Estimated Reading Time: 10 min
Difficulty: Advanced
Theory Content: app/service/repository/model layering for maintainability.
Real World Analogy: specialized departments in a company.
Visual Diagram (Markdown): CLI -> Service -> Repository -> Storage
Expected Output: maintainable package structure.
Common Mistakes: business logic in UI layer.
Best Practices: one responsibility per class.
Mini Exercise: draft package map before coding.
Key Takeaways: architecture reduces rewrite pain.`
            },
            {
                title: "Lesson 3: Integrating OOP, Collections, and Exceptions",
                content: `Learning Objective: unify previous modules in cohesive flows.
Estimated Reading Time: 12 min
Difficulty: Advanced
Theory Content: model entities, store data with collections, enforce business rules with exceptions.
Real World Analogy: operations center with validation checkpoints.
Visual Diagram (Markdown): entity -> service rule -> collection store -> response
Expected Output: safe business operations.
Common Mistakes: bypassing validation path.
Best Practices: centralize validation rules.
Mini Exercise: add duplicate-prevention rule using map/set.
Key Takeaways: integration quality is capstone core.`
            },
            {
                title: "Lesson 4: File Storage and Data Lifecycle",
                content: `Learning Objective: persist and reload application state.
Estimated Reading Time: 9 min
Difficulty: Advanced
Theory Content: serialize/write/read records with recovery paths for corrupt data.
Real World Analogy: ledger books with backup copies.
Visual Diagram (Markdown): in-memory state <-> file persistence
Expected Output: restart-safe application state.
Common Mistakes: no backup or invalid format handling.
Best Practices: version storage format and validate on load.
Mini Exercise: write import/export commands.
Key Takeaways: persistence defines real-world usability.`
            },
            {
                title: "Lesson 5: Testing, Documentation, and Demo Readiness",
                content: `Learning Objective: prepare project for evaluation and interviews.
Estimated Reading Time: 9 min
Difficulty: Advanced
Theory Content: test scenarios, usage docs, and demo script improve project credibility.
Real World Analogy: final rehearsal before product launch.
Visual Diagram (Markdown): test plan -> fixes -> demo -> review
Expected Output: stable and explainable project.
Common Mistakes: skipping edge case tests.
Best Practices: include setup, run, and assumptions in README.
Mini Exercise: write 5 test scenarios including failure case.
Key Takeaways: communication quality matters as much as code.`
            },
            {
                title: "Suggested Project Options",
                content: `Pick one capstone based on your target role.

Banking System
• Best for: backend and transaction-heavy logic practice.
• Build core: create account, deposit/withdraw, transfer, transaction history.
• Add one stretch: monthly statement export or fraud alert rules.

Library Management System
• Best for: CRUD workflows and data lifecycle handling.
• Build core: add books, issue/return flow, member records, overdue tracking.
• Add one stretch: fine calculator or waitlist queue.

Inventory Management System
• Best for: operations and reporting scenarios.
• Build core: item catalog, stock in/out, low-stock alerts, supplier info.
• Add one stretch: reorder recommendation or CSV import/export.

Expense Tracker
• Best for: clean UX flow and analytics basics.
• Build core: add expense, category filter, monthly summary, budget warnings.
• Add one stretch: trend report or recurring expense automation.

Student Management System
• Best for: object modeling and validation-heavy rules.
• Build core: student profile, attendance, marks, grade report.
• Add one stretch: top-performer ranking or parent progress snapshot.`
            },
            {
                title: "Project Deliverables",
                content: `Your final submission must include these 4 deliverable packs.

Code Pack:
• clean layered architecture (CLI -> service -> repository -> storage)
• OOP models with clear responsibilities
• collections used intentionally (List/Map/Set)
• robust exception handling with user-friendly messages

Persistence Pack:
• file-based save/load for app state
• invalid/corrupt data handling path
• no data loss on normal restart flow

Quality Pack:
• at least 10 manual test scenarios (including edge/failure cases)
• modular methods with readable names
• no repeated logic blocks across services

Documentation Pack:
• README with setup, run steps, and feature list
• architecture explanation with trade-offs
• 5-minute demo script for interview walkthrough`
            },
            {
                title: "Final Assessment",
                content: `Purpose: this tab is your job-readiness checkpoint, not just another quiz.
Why it exists: it verifies that you can explain, build, and defend your capstone like in real interviews.

Assessment Format:
• Concept Check: MCQs from Modules 1-6 (syntax, OOP, collections, exceptions, file handling, concurrency basics)
• Build Check: timed coding challenges using your capstone patterns
• Debug Check: identify and fix bugs in a broken Java workflow
• Interview Check: answer architecture and trade-off questions verbally/in writing

Pass Criteria:
• concept clarity and correct reasoning
• working and modular code submission
• clear explanation of design decisions
• confidence in project walkthrough`
            },
            {
                title: "Mini Challenge",
                content: `Implement one stretch feature (search, report export, role permissions, or analytics) without breaking existing flow.`
            },
            {
                title: "Interview Preparation",
                content: `Prepare to answer:
• why this architecture?
• trade-offs made and why
• how errors are handled end-to-end
• scaling plan for persistence and multi-user support
• demo walkthrough in under 5 minutes`
            },
            {
                title: "Module Summary",
                content: `You now have a complete Java project and assessment path aligned to job-ready expectations.`
            },
            {
                title: "Course Completion Path",
                content: `Complete final assessment, polish capstone README, and prepare project walkthrough for interview portfolio.`
            }
        ]
    }
];

courseData.pythonProgramming = [
    {
        number: "Module 1",
        title: "Python Fundamentals",
        description: "Build strong Python foundations from setup to core syntax, control flow, functions, and built-in collections.",
        duration: "60 min",
        lessons: "12 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Introduction to Python", "Installing Python & IDE Setup", "Running Python Programs", "Python Syntax", "Variables", "Data Types", "Operators", "User Input", "Type Casting", "Conditional Statements", "Loops", "Functions", "Modules", "Packages", "Strings", "Lists", "Tuples", "Sets", "Dictionaries"],
        detailedDescription: "Foundational module for writing clean Python code and understanding core language constructs used in every track.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 1
Module Name: Python Fundamentals
Difficulty: Beginner
Estimated Reading Time: 60 min
Estimated Completion Time: 6-8 hours
Prerequisites: None
Learning Objectives:
• set up Python environment and run scripts confidently
• write syntax-correct programs using variables, conditions, loops, and functions
• use Python core collections for real data manipulation
Skills Gained:
• Python execution model and syntax confidence
• control flow and reusable function design
• practical data structure usage`
            },
            {
                title: "Lesson 1: Setup, Installation, and Running Python",
                content: `Learning Objective: set up a stable Python development environment.
Estimated Reading Time: 10 min
Difficulty: Beginner
Theory Content: install Python, configure IDE, and run scripts from terminal and editor.
Real World Analogy: setting up a workshop before building products.
Visual Diagram (Markdown): install -> interpreter -> script run -> output
Expected Output: successful first Python run on local machine.
Common Mistakes: version mismatch and PATH issues.
Best Practices: verify with python --version and isolate projects early.
Mini Exercise: run hello script from IDE and terminal.
Key Takeaways: reliable setup prevents avoidable learning friction.`
            },
            {
                title: "Lesson 2: Syntax, Variables, Data Types, and Operators",
                content: `Learning Objective: write syntactically correct expressions and variable logic.
Estimated Reading Time: 12 min
Difficulty: Beginner
Theory Content: Python indentation, naming rules, primitive types, and arithmetic/logical operators.
Real World Analogy: grammar and vocabulary rules in a language.
Visual Diagram (Markdown): input values -> operators -> computed result
Expected Output: correct expressions and type-aware assignments.
Common Mistakes: mixing incompatible types and indentation errors.
Best Practices: keep expressions explicit and readable.
Mini Exercise: calculate invoice totals with tax and discount.
Key Takeaways: syntax discipline is the base of all later modules.`
            },
            {
                title: "Lesson 3: Input, Type Casting, Conditionals, and Loops",
                content: `Learning Objective: build interactive and decision-driven Python programs.
Estimated Reading Time: 12 min
Difficulty: Beginner
Theory Content: user input handling, safe type conversion, branching, and iterative processing.
Real World Analogy: form intake and rule engine for approvals.
Visual Diagram (Markdown): input -> cast -> condition -> loop -> result
Expected Output: validated input flow and branch logic.
Common Mistakes: unsafe casting and infinite loops.
Best Practices: validate user input before conversion.
Mini Exercise: build grade evaluator from entered marks.
Key Takeaways: data validation and flow control improve reliability.`
            },
            {
                title: "Lesson 4: Functions, Modules, and Packages",
                content: `Learning Objective: organize code into reusable components.
Estimated Reading Time: 12 min
Difficulty: Beginner
Theory Content: function arguments/returns, import system, and package structure basics.
Real World Analogy: reusable tools organized in labeled drawers.
Visual Diagram (Markdown): function -> module -> package -> app
Expected Output: split code across modules and call imported functions.
Common Mistakes: circular imports and oversized functions.
Best Practices: keep modules focused on one responsibility.
Mini Exercise: create utility module and import from main script.
Key Takeaways: modular structure supports scale and testing.`
            },
            {
                title: "Lesson 5: Strings and Core Collections",
                content: `Learning Objective: solve data tasks with strings, lists, tuples, sets, and dictionaries.
Estimated Reading Time: 14 min
Difficulty: Beginner
Theory Content: each collection's strengths for ordering, uniqueness, and lookup.
Real World Analogy: toolbox where each compartment serves a specific need.
Visual Diagram (Markdown): raw text -> split -> collection transform -> output
Expected Output: cleaner data transformation logic.
Common Mistakes: using wrong collection for lookup-heavy tasks.
Best Practices: choose data structures by operation cost.
Mini Exercise: parse sentence and output word frequency map.
Key Takeaways: collection choice directly affects code quality.`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
• build StudentResultConsole using input, casting, conditionals, loops, and functions
• include collection-based storage for subject marks and average calculation

Success Check:
• grade logic is correct for all boundary values
• output summary is clear, readable, and formatted consistently`
            },
            {
                title: "Mini Assessment",
                content: `1) Python blocks are identified by: A) braces B) indentation C) semicolon D) compiler flags
2) Best way to reuse logic is: A) copy-paste B) function C) global variable D) recursion only
3) Which is mutable? A) tuple B) str C) list D) int
4) Type casting is used to: A) import module B) convert data type C) run loop D) raise exception
5) Best structure for key lookup: A) list B) dict C) tuple D) set`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
• explain Python execution flow from script to output
• mutable vs immutable with practical examples
• loops vs comprehensions trade-off
• function design for readability and testability`
            },
            {
                title: "Module Summary",
                content: `You can now read, write, and run Python confidently with clean syntax, control flow, reusable functions, and core collections.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module moves from procedural coding to object-oriented design with classes, inheritance, and abstraction.`
            }
        ]
    },
    {
        number: "Module 2",
        title: "Object-Oriented Programming in Python",
        description: "Master OOP foundations in Python and apply them in a practical Library Management System mini project.",
        duration: "70 min",
        lessons: "12 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Classes", "Objects", "Constructors (__init__)", "Instance vs Class Variables", "Instance vs Static Methods", "Encapsulation", "Inheritance", "Polymorphism", "Abstraction", "Magic (Dunder) Methods", "Properties", "Dataclasses", "Packages & Imports"],
        detailedDescription: "Core OOP module for writing maintainable Python applications with clear class design and object collaboration.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 2
Module Name: Object-Oriented Programming in Python
Difficulty: Intermediate
Estimated Reading Time: 70 min
Estimated Completion Time: 7-9 hours
Prerequisites: Module 1
Learning Objectives:
• model real entities using classes and objects
• apply inheritance, polymorphism, and abstraction responsibly
• design clean OOP architecture with imports and packages
Skills Gained:
• class and object lifecycle design
• reusable OOP patterns in Python
• maintainable project structure for medium systems`
            },
            {
                title: "Lesson 1: Classes, Objects, and Constructors",
                content: `Learning Objective: create reliable object blueprints with __init__.
Estimated Reading Time: 12 min
Difficulty: Intermediate
Theory Content: class design, object instantiation, and constructor-based state setup.
Real World Analogy: blueprint-to-building process.
Visual Diagram (Markdown): class -> object instances -> method calls
Expected Output: predictable object state and behavior.
Common Mistakes: missing initialization and unclear attributes.
Best Practices: keep constructors explicit and focused.
Mini Exercise: create Book class with constructor and display method.
Key Takeaways: constructor quality sets design quality.`
            },
            {
                title: "Lesson 2: Variables, Methods, and Encapsulation",
                content: `Learning Objective: choose between instance/class state and method types correctly.
Estimated Reading Time: 12 min
Difficulty: Intermediate
Theory Content: instance vs class variables, instance vs static methods, private/protected conventions.
Real World Analogy: personal account data vs company-wide policy.
Visual Diagram (Markdown): object state + class state + method access
Expected Output: cleaner class behavior boundaries.
Common Mistakes: overusing class variables for mutable state.
Best Practices: use static methods for stateless utility logic.
Mini Exercise: add class-level ID counter to Member class.
Key Takeaways: state placement affects correctness and maintainability.`
            },
            {
                title: "Lesson 3: Inheritance, Polymorphism, and Abstraction",
                content: `Learning Objective: extend behavior without duplication.
Estimated Reading Time: 14 min
Difficulty: Intermediate
Theory Content: base classes, overrides, abstract contracts, and dynamic dispatch.
Real World Analogy: shared company policy with role-specific behavior.
Visual Diagram (Markdown): base class -> child classes -> polymorphic call
Expected Output: extensible object hierarchies.
Common Mistakes: inheritance where composition is better.
Best Practices: keep base classes minimal and meaningful.
Mini Exercise: create LoanPolicy base class with child rules.
Key Takeaways: polymorphism enables flexible scaling.`
            },
            {
                title: "Lesson 4: Dunder Methods, Properties, and Dataclasses",
                content: `Learning Objective: improve class usability and readability.
Estimated Reading Time: 12 min
Difficulty: Intermediate
Theory Content: __str__, __repr__, comparison dunders, @property, and dataclass usage.
Real World Analogy: adding clear labels and auto-generated forms to records.
Visual Diagram (Markdown): raw class -> dunders/properties -> cleaner API
Expected Output: more maintainable and expressive class interfaces.
Common Mistakes: exposing mutable internals directly.
Best Practices: use dataclass for data carriers and property for controlled access.
Mini Exercise: convert DTO class into dataclass.
Key Takeaways: Pythonic classes reduce boilerplate and bugs.`
            },
            {
                title: "Lesson 5: Packages and Imports for OOP Projects",
                content: `Learning Objective: structure OOP codebase for growth.
Estimated Reading Time: 10 min
Difficulty: Intermediate
Theory Content: package hierarchy, absolute/relative imports, and modular organization.
Real World Analogy: organizing departments in a company directory.
Visual Diagram (Markdown): package -> module -> class -> usage
Expected Output: import-safe, maintainable project layout.
Common Mistakes: circular imports and flat monolithic files.
Best Practices: split model/service/repository layers.
Mini Exercise: refactor class files into package modules.
Key Takeaways: structure quality determines long-term speed.`
            },
            {
                title: "Mini Project: Library Management System",
                content: `Project Goal: build an OOP-first library system using classes, inheritance, and modular design.

Domain Modeling:
• entities: Book, Member, Loan, Librarian
• constructor-based initialization and validation
• dataclass for read-only view models where helpful

Core Workflow:
• add/search books and register members
• issue/return books with rule checks
• maintain borrowing history and availability state

Architecture Expectations:
• package structure for model/service/app modules
• encapsulated class behavior with clean interfaces
• use dunder methods/properties for better readability

Evaluation Signals:
• clear OOP responsibilities and low coupling
• correct issue/return constraints
• maintainable class hierarchy and imports`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
• add overdue fee strategy and reservation queue without breaking existing interfaces
• expose member summary with active loans and pending reservations

Success Check:
• new features integrate cleanly with existing class design
• issue/return behavior remains correct for old and new scenarios`
            },
            {
                title: "Module Quiz",
                content: `1) __init__ is used for: A) import B) object initialization C) loop D) type conversion
2) Best use of @property is: A) random print B) controlled attribute access C) threading D) package install
3) Polymorphism means: A) one class only B) same interface, different implementations C) no inheritance D) no methods
4) Dataclass is ideal for: A) heavy business service B) data carrier model C) network socket D) CLI parser
5) Good package design emphasizes: A) circular imports B) cohesive modules C) giant file D) duplicated logic`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
• composition vs inheritance trade-offs
• class vs static method decision-making
• abstraction in Python without overengineering
• debugging circular imports and class design smells`
            },
            {
                title: "Module Summary",
                content: `You can now design Python OOP systems with maintainable classes, clear abstractions, and package-level organization.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module deepens Python with comprehensions, iterators, generators, robust exceptions, and file handling.`
            }
        ]
    },
    {
        number: "Module 3",
        title: "Advanced Python & Error Handling",
        description: "Master advanced data handling, comprehensions, generators, exception patterns, and file processing with a Student Record Manager mini project.",
        duration: "75 min",
        lessons: "12 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Collections & Advanced Data Structures", "List Comprehensions", "Dictionary Comprehensions", "Set Comprehensions", "Iterators", "Generators", "try", "except", "else", "finally", "raise", "Custom Exceptions", "Reading Files", "Writing Files", "CSV Files", "JSON Files"],
        detailedDescription: "Advanced Python module focused on expressive data pipelines and resilient exception handling for real workloads.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 3
Module Name: Advanced Python & Error Handling
Difficulty: Intermediate
Estimated Reading Time: 75 min
Estimated Completion Time: 8-10 hours
Prerequisites: Modules 1-2
Learning Objectives:
• write expressive transforms with comprehensions and generators
• design reliable error handling with custom exceptions
• process CSV/JSON/files safely for practical workflows
Skills Gained:
• advanced iteration and memory-aware processing
• structured exception design and recovery
• robust file and data pipeline handling`
            },
            {
                title: "Lesson 1: Advanced Collections and Comprehensions",
                content: `Learning Objective: transform datasets concisely with comprehensions.
Estimated Reading Time: 12 min
Difficulty: Intermediate
Theory Content: list/dict/set comprehensions with conditions and mapping logic.
Real World Analogy: assembly line sorting and reshaping products.
Visual Diagram (Markdown): input collection -> filter/map -> output collection
Expected Output: concise, readable transforms.
Common Mistakes: nested comprehensions that reduce readability.
Best Practices: prefer explicit loops when complexity grows.
Mini Exercise: build summary dict from student score list.
Key Takeaways: concise code must remain understandable.`
            },
            {
                title: "Lesson 2: Iterators and Generators",
                content: `Learning Objective: process large data streams efficiently.
Estimated Reading Time: 12 min
Difficulty: Intermediate
Theory Content: iterator protocol, lazy generator execution, and yield-based pipelines.
Real World Analogy: on-demand conveyor instead of storing all items in warehouse.
Visual Diagram (Markdown): source -> iterator -> generator -> consumer
Expected Output: lower memory usage for large input flows.
Common Mistakes: exhausting iterators unintentionally.
Best Practices: document one-pass consumption behavior.
Mini Exercise: create generator for streaming valid rows only.
Key Takeaways: laziness improves scalability.`
            },
            {
                title: "Lesson 3: Exception Handling Patterns",
                content: `Learning Objective: handle failure scenarios predictably.
Estimated Reading Time: 12 min
Difficulty: Intermediate
Theory Content: try/except/else/finally, raise, and domain-specific custom exceptions.
Real World Analogy: incident handling workflow with categorized tickets.
Visual Diagram (Markdown): risky operation -> exception mapping -> recovery/log
Expected Output: safer and debuggable error paths.
Common Mistakes: broad except that hides root cause.
Best Practices: catch specific exceptions and enrich context.
Mini Exercise: create InvalidRecordError with source metadata.
Key Takeaways: explicit errors improve maintainability.`
            },
            {
                title: "Lesson 4: File Handling with CSV and JSON",
                content: `Learning Objective: read and write structured files safely.
Estimated Reading Time: 12 min
Difficulty: Intermediate
Theory Content: text file operations, csv module, json load/dump, and encoding practices.
Real World Analogy: receiving forms, validating, and archiving with standard format.
Visual Diagram (Markdown): file input -> parse -> validate -> write output
Expected Output: robust import/export flow.
Common Mistakes: ignoring encoding and malformed rows.
Best Practices: use context managers and schema validation.
Mini Exercise: parse CSV and emit JSON summary.
Key Takeaways: file discipline is essential in production scripts.`
            },
            {
                title: "Lesson 5: Integrated Processing Pipeline",
                content: `Learning Objective: combine generators, exceptions, and file handlers into one workflow.
Estimated Reading Time: 12 min
Difficulty: Intermediate
Theory Content: staged pipeline design for parse, validate, transform, and report.
Real World Analogy: warehouse intake with quality gates and exception bins.
Visual Diagram (Markdown): source file -> parser -> validator -> aggregator -> report
Expected Output: deterministic output with reject tracking.
Common Mistakes: mixed concerns and silent failure.
Best Practices: isolate each stage in testable functions.
Mini Exercise: add rejected row summary to final report.
Key Takeaways: composable pipelines are scalable and testable.`
            },
            {
                title: "Mini Project: Student Record Manager",
                content: `Project Goal: build a reliable record manager with advanced collections and robust exception handling.

Data Design:
• maintain score history and metadata with nested structures
• use comprehensions for fast summary generation
• stream large input with generators where useful

Reliability Requirements:
• custom exceptions for invalid marks and malformed rows
• explicit try/except/else/finally paths for file operations
• no partial corruption when errors occur

File Handling:
• import CSV records and export JSON/CSV summary reports
• log rejected records with clear reason code
• keep parsing and validation as separate layers

Evaluation Signals:
• accurate summaries across edge cases
• clear error categorization and recovery
• maintainable function-level pipeline structure`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
• add iterator-based chunk processing for very large files
• include retry-safe read path for transient file access issues

Success Check:
• pipeline processes valid rows even when malformed rows exist
• summary report includes processed, rejected, and retried counts`
            },
            {
                title: "Module Quiz",
                content: `1) Generator advantage is: A) eager load B) lazy processing C) no iteration D) fixed memory spike
2) finally block is used for: A) optional comment B) guaranteed cleanup C) import D) cast
3) Best response to malformed row is: A) crash always B) reject with reason C) ignore silently D) overwrite
4) Custom exception helps with: A) syntax color B) domain clarity C) package install D) speed only
5) JSON module is used for: A) threading B) serialization C) encryption D) plotting`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
• iterators vs generators with memory trade-offs
• exception boundary design (raise vs catch)
• designing robust CSV/JSON ingestion pipeline
• ensuring deterministic behavior under bad input`
            },
            {
                title: "Module Summary",
                content: `You can now build advanced Python pipelines with expressive transforms, resilient exceptions, and robust file workflows.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module introduces modern Python techniques like lambda pipelines, decorators, context managers, and packaging workflows.`
            }
        ]
    },
    {
        number: "Module 4",
        title: "Modern Python Programming",
        description: "Adopt modern Python patterns for functional transforms, decorators, context managers, typing, logging, and packaging.",
        duration: "70 min",
        lessons: "12 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Lambda Functions", "map()", "filter()", "reduce()", "Decorators", "Closures", "Context Managers", "Type Hints", "Enums", "Named Tuples", "Virtual Environments", "pip", "Python Packaging", "Logging", "Configuration Management"],
        detailedDescription: "Modern Python module focused on readable functional patterns, stronger typing, and production-friendly tooling.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 4
Module Name: Modern Python Programming
Difficulty: Intermediate
Estimated Reading Time: 70 min
Estimated Completion Time: 7-9 hours
Prerequisites: Modules 1-3
Learning Objectives:
• write concise but readable functional Python code
• use decorators, context managers, and type hints effectively
• package and configure Python applications for repeatable delivery
Skills Gained:
• functional and declarative coding fluency
• observability with logging and config management
• environment and packaging hygiene`
            },
            {
                title: "Lesson 1: Lambda, map, filter, reduce",
                content: `Learning Objective: build concise transformation pipelines.
Estimated Reading Time: 12 min
Difficulty: Intermediate
Theory Content: anonymous functions and functional helpers for map/filter/reduce operations.
Real World Analogy: assembly line operations with staged transformations.
Visual Diagram (Markdown): data -> map -> filter -> reduce -> result
Expected Output: concise aggregation and transformation logic.
Common Mistakes: overusing lambda where named function is clearer.
Best Practices: prioritize readability over compactness.
Mini Exercise: compute department-wise salary totals with reduce.
Key Takeaways: functional tools are powerful when used intentionally.`
            },
            {
                title: "Lesson 2: Decorators, Closures, and Context Managers",
                content: `Learning Objective: manage cross-cutting behavior and resources cleanly.
Estimated Reading Time: 14 min
Difficulty: Intermediate
Theory Content: decorator patterns, closure scopes, and context-managed resource safety.
Real World Analogy: automated checkpoints before and after every operation.
Visual Diagram (Markdown): call -> decorator wrapper -> function -> cleanup
Expected Output: reusable timing/logging wrappers and safe file/resource handling.
Common Mistakes: hidden side effects in decorators.
Best Practices: keep wrappers transparent and testable.
Mini Exercise: add execution-time decorator to analytics function.
Key Takeaways: abstraction should improve clarity, not hide behavior.`
            },
            {
                title: "Lesson 3: Type Hints, Enums, and Named Tuples",
                content: `Learning Objective: improve code contracts and model clarity.
Estimated Reading Time: 12 min
Difficulty: Intermediate
Theory Content: static type hints, constrained values with Enum, lightweight records with named tuples.
Real World Analogy: standardized labels and forms for consistent data handling.
Visual Diagram (Markdown): typed inputs -> validated model -> safer function use
Expected Output: clearer function signatures and safer domain modeling.
Common Mistakes: type hints with inaccurate contracts.
Best Practices: annotate public interfaces first.
Mini Exercise: define typed analytics result model.
Key Takeaways: explicit contracts reduce integration bugs.`
            },
            {
                title: "Lesson 4: Virtual Environments, pip, and Packaging",
                content: `Learning Objective: package Python projects for repeatable execution.
Estimated Reading Time: 12 min
Difficulty: Intermediate
Theory Content: environment isolation, dependency pinning, and package metadata basics.
Real World Analogy: shipping kit with exact parts list.
Visual Diagram (Markdown): source -> venv -> requirements -> package install
Expected Output: reproducible setup for any contributor.
Common Mistakes: global installs and unpinned dependencies.
Best Practices: document install/run commands in README.
Mini Exercise: create venv and export dependency file.
Key Takeaways: reproducibility is a production requirement.`
            },
            {
                title: "Lesson 5: Logging and Configuration Management",
                content: `Learning Objective: make Python applications observable and configurable.
Estimated Reading Time: 10 min
Difficulty: Intermediate
Theory Content: logging levels, structured logs, and environment/config-driven behavior.
Real World Analogy: control room dashboard and operating playbook.
Visual Diagram (Markdown): runtime events -> logger -> file/console -> diagnostics
Expected Output: context-rich logs and environment-aware behavior.
Common Mistakes: logging secrets and hardcoded constants.
Best Practices: centralize config loading and redact sensitive values.
Mini Exercise: add env-based config for log level.
Key Takeaways: observability and config control speed up debugging.`
            },
            {
                title: "Mini Project: Employee Analytics Dashboard",
                content: `Project Goal: build a modern analytics dashboard backend/CLI with functional transforms and typed outputs.

Data Pipeline:
• ingest employee records and normalize fields
• transform datasets with map/filter/reduce workflows
• generate grouped analytics with clear KPI summaries

Engineering Requirements:
• add type hints for core services
• enforce constrained states using Enum models
• use context managers for file/report lifecycle safety

Operations Requirements:
• structured logging for each pipeline stage
• config-driven thresholds and output paths
• package-ready project layout with dependency file

Evaluation Signals:
• readable transform logic and typed interfaces
• stable output across sample datasets
• strong logging and configurable behavior`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
• add anomaly detection layer with decorator-based performance tracing
• export dashboard summary to both JSON and CSV formats

Success Check:
• anomaly flags match expected sample scenarios
• logs and output files remain consistent across reruns`
            },
            {
                title: "Module Quiz",
                content: `1) reduce() is mainly for: A) sorting B) aggregation C) threading D) packaging
2) Decorator is used to: A) replace imports B) wrap behavior C) cast types D) create package
3) Context manager ensures: A) no loops B) deterministic setup/cleanup C) no errors D) API call
4) Type hints primarily improve: A) runtime speed only B) readability and tooling checks C) syntax coloring D) packaging size
5) venv helps with: A) global state sharing B) dependency isolation C) logging format D) class inheritance`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
• lambda vs named function trade-offs
• decorator design for logging and timing
• config management in multi-environment deployments
• packaging and dependency reproducibility practices`
            },
            {
                title: "Module Summary",
                content: `You can now write modern Python with functional tools, stronger typing, robust configuration, and packaging discipline.`
            },
            {
                title: "Next Module Bridge",
                content: `Next module adds concurrency, automation tooling, API integration, testing, and production optimization.`
            }
        ]
    },
    {
        number: "Module 5",
        title: "Concurrency, Automation & Production Python",
        description: "Learn concurrency models, automation tooling, API integration, testing, and performance optimization for production Python.",
        duration: "80 min",
        lessons: "12 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Threads", "Multiprocessing", "Async Programming", "asyncio", "OS Module", "shutil", "pathlib", "subprocess", "Requests Library", "REST API Consumption", "Environment Variables", "Configuration Files", "Unit Testing", "Debugging", "Performance Optimization"],
        detailedDescription: "Production-focused module combining scalable execution, automation scripts, external APIs, and quality practices.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 5
Module Name: Concurrency, Automation & Production Python
Difficulty: Intermediate to Advanced
Estimated Reading Time: 80 min
Estimated Completion Time: 8-10 hours
Prerequisites: Modules 1-4
Learning Objectives:
• choose the right concurrency model for workloads
• automate file/system tasks with Python stdlib tools
• ship production-ready code with tests, debugging, and performance tuning
Skills Gained:
• thread/process/async decision-making
• automation workflows with OS and subprocess tooling
• resilient API integration and test coverage`
            },
            {
                title: "Lesson 1: Threads, Multiprocessing, and asyncio",
                content: `Learning Objective: select concurrency approach based on CPU/IO profile.
Estimated Reading Time: 14 min
Difficulty: Intermediate to Advanced
Theory Content: threading for I/O, multiprocessing for CPU work, and asyncio for async orchestration.
Real World Analogy: assigning workers, machines, and dispatch lines by task type.
Visual Diagram (Markdown): workload type -> concurrency model -> executor loop
Expected Output: improved throughput with controlled complexity.
Common Mistakes: using threads for heavy CPU tasks under GIL constraints.
Best Practices: benchmark before and after concurrency changes.
Mini Exercise: compare threaded vs async download workflow.
Key Takeaways: model selection matters more than premature optimization.`
            },
            {
                title: "Lesson 2: Automation with os, shutil, pathlib, subprocess",
                content: `Learning Objective: automate filesystem and command workflows safely.
Estimated Reading Time: 14 min
Difficulty: Intermediate
Theory Content: path abstraction, file movement, process invocation, and script orchestration.
Real World Analogy: warehouse conveyor and routing automation.
Visual Diagram (Markdown): input folder -> classify -> move -> external command -> report
Expected Output: reproducible automation script.
Common Mistakes: string path manipulation and unsafe shell calls.
Best Practices: use pathlib and validate command exit statuses.
Mini Exercise: organize files by extension and generate summary.
Key Takeaways: automation scripts need safety and observability.`
            },
            {
                title: "Lesson 3: Requests and REST API Consumption",
                content: `Learning Objective: consume external APIs reliably.
Estimated Reading Time: 12 min
Difficulty: Intermediate
Theory Content: request lifecycle, timeout/retry strategy, status handling, and response validation.
Real World Analogy: supplier communication with response verification.
Visual Diagram (Markdown): request -> response -> validate -> transform -> store
Expected Output: fault-tolerant API client behavior.
Common Mistakes: no timeout and unchecked JSON schema assumptions.
Best Practices: isolate API client and handle retriable/non-retriable errors separately.
Mini Exercise: build API client with timeout and fallback response.
Key Takeaways: reliable integrations require defensive coding.`
            },
            {
                title: "Lesson 4: Environment, Config, and Quality Tooling",
                content: `Learning Objective: manage environment-specific runtime behavior safely.
Estimated Reading Time: 10 min
Difficulty: Intermediate
Theory Content: environment variables, config files, and runtime validation of required settings.
Real World Analogy: machine operating modes for different factories.
Visual Diagram (Markdown): env vars + config file -> validator -> runtime settings
Expected Output: predictable multi-environment execution.
Common Mistakes: hardcoded secrets and missing config defaults.
Best Practices: centralize config loading and fail fast on critical gaps.
Mini Exercise: add settings loader with required-key checks.
Key Takeaways: config discipline prevents deployment incidents.`
            },
            {
                title: "Lesson 5: Unit Testing, Debugging, and Performance",
                content: `Learning Objective: verify correctness and optimize bottlenecks responsibly.
Estimated Reading Time: 12 min
Difficulty: Intermediate
Theory Content: unit testing strategy, debugger workflow, profiling basics, and optimization checkpoints.
Real World Analogy: quality lab and diagnostic station for production line.
Visual Diagram (Markdown): code -> test -> debug -> profile -> optimize
Expected Output: stable logic with measurable performance improvements.
Common Mistakes: optimization before baseline measurement.
Best Practices: test first, then optimize hotspots.
Mini Exercise: profile and optimize one slow function.
Key Takeaways: disciplined iteration beats guesswork.`
            },
            {
                title: "Mini Project: Automated File Organizer",
                content: `Project Goal: build an automation utility that classifies, organizes, and reports on files reliably.

Automation Workflow:
• scan directories with pathlib
• classify/move files using rules with shutil and os
• execute optional post-actions through subprocess hooks

Production Requirements:
• add concurrency for large directory processing
• call external metadata API for enrichment where configured
• configure behavior through env vars/config file

Quality Requirements:
• unit tests for routing rules and conflict handling
• debug logs for each move decision
• performance metrics for large batch runs

Evaluation Signals:
• deterministic organization output
• safe handling of collisions and errors
• test-backed and observable workflow`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
• add async API enrichment stage with retry and timeout control
• include dry-run mode and rollback report for moved files

Success Check:
• dry-run predicts actions accurately before execution
• rollback log is sufficient to restore moved files safely`
            },
            {
                title: "Module Quiz",
                content: `1) Best fit for CPU-bound tasks: A) threading B) multiprocessing C) asyncio only D) no concurrency
2) pathlib mainly improves: A) GUI B) path handling readability and safety C) JSON parsing D) unit tests
3) requests timeout should be: A) omitted B) explicitly set C) randomized D) infinite
4) Unit tests should cover: A) happy path only B) edge and failure cases too C) imports only D) comments
5) Performance optimization should start after: A) guess B) profiling baseline C) deployment D) refactor all code`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
• threads vs async vs multiprocessing use-cases
• resilient API client design patterns
• environment config and secrets handling
• test strategy for automation workflows`
            },
            {
                title: "Module Summary",
                content: `You can now build production-minded Python automation with concurrency, API safety, testing rigor, and measurable performance.`
            },
            {
                title: "Next Module Bridge",
                content: `Final module combines all prior learning into one complete real-world Python capstone project.`
            }
        ]
    },
    {
        number: "Module 6",
        title: "Real-World Python Capstone Project",
        description: "Build and present a complete real-world Python application with architecture, testing, documentation, and interview readiness.",
        duration: "90 min",
        lessons: "12 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Capstone architecture", "OOP integration", "File handling", "Exception handling", "Modular code", "Logging", "Configuration management", "Documentation", "Unit tests", "Suggested projects", "Project deliverables", "Final assessment"],
        detailedDescription: "Capstone module to consolidate all Python learning into a portfolio-grade project with assessment-driven job readiness.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 6
Module Name: Real-World Python Capstone Project
Difficulty: Advanced
Estimated Reading Time: 90 min
Estimated Completion Time: 10-12 hours
Prerequisites: Modules 1-5
Learning Objectives:
• architect and deliver a complete Python application
• integrate OOP, robust error handling, file processing, logging, and config
• present project with tests and documentation for interviews
Skills Gained:
• end-to-end project execution
• production-ready coding standards
• interview-ready project communication`
            },
            {
                title: "Lesson 1: Project Scoping and Requirements",
                content: `Learning Objective: define realistic capstone scope and milestones.
Estimated Reading Time: 12 min
Difficulty: Advanced
Theory Content: identify core entities, workflows, constraints, and MVP boundaries.
Real World Analogy: creating a product blueprint before build.
Visual Diagram (Markdown): requirements -> modules -> milestones
Expected Output: crisp capstone scope document.
Common Mistakes: overscoping first version.
Best Practices: prioritize MVP and defer nice-to-have features.
Mini Exercise: define acceptance criteria for one suggested project.
Key Takeaways: scope quality controls delivery quality.`
            },
            {
                title: "Lesson 2: Architecture and Modular Implementation",
                content: `Learning Objective: organize code into maintainable layers.
Estimated Reading Time: 12 min
Difficulty: Advanced
Theory Content: app/service/repository/model layering and dependency boundaries.
Real World Analogy: specialized teams with clear responsibilities.
Visual Diagram (Markdown): CLI/API -> Service -> Repository -> Storage
Expected Output: maintainable and test-friendly project structure.
Common Mistakes: business logic mixed with UI/CLI concerns.
Best Practices: one responsibility per module.
Mini Exercise: draft module map before coding.
Key Takeaways: architecture clarity lowers maintenance cost.`
            },
            {
                title: "Lesson 3: Validation, Exceptions, and Persistence",
                content: `Learning Objective: build resilient workflows under imperfect input.
Estimated Reading Time: 12 min
Difficulty: Advanced
Theory Content: validation gates, custom exceptions, and file/data persistence strategy.
Real World Analogy: quality control and audit logging in operations.
Visual Diagram (Markdown): input -> validate -> process -> persist -> report
Expected Output: stable behavior with meaningful failures.
Common Mistakes: silent exception handling.
Best Practices: include context-rich error messages.
Mini Exercise: implement reject log with reason codes.
Key Takeaways: resilient systems fail gracefully.`
            },
            {
                title: "Lesson 4: Testing, Logging, and Configuration",
                content: `Learning Objective: ensure quality and observability.
Estimated Reading Time: 12 min
Difficulty: Advanced
Theory Content: unit testing strategy, logging standards, and environment-configured behavior.
Real World Analogy: QA lab and control panel for operations.
Visual Diagram (Markdown): tests + logs + config -> stable deployment
Expected Output: predictable and diagnosable application behavior.
Common Mistakes: no failure-path test coverage.
Best Practices: cover happy, edge, and failure scenarios.
Mini Exercise: write tests for one critical service path.
Key Takeaways: quality tooling increases delivery confidence.`
            },
            {
                title: "Lesson 5: Documentation and Demo Readiness",
                content: `Learning Objective: present capstone effectively for evaluation.
Estimated Reading Time: 10 min
Difficulty: Advanced
Theory Content: README structure, run guide, architecture notes, and demo walkthrough script.
Real World Analogy: launch-day runbook and stakeholder demo prep.
Visual Diagram (Markdown): docs -> demo flow -> interview narrative
Expected Output: interview-ready project presentation artifacts.
Common Mistakes: undocumented assumptions and setup gaps.
Best Practices: include sample input/output and quick-start steps.
Mini Exercise: draft 5-minute demo narrative.
Key Takeaways: communication is part of engineering quality.`
            },
            {
                title: "Suggested Project Options",
                content: `Choose one capstone aligned to your career goals.

Expense Tracker
• Build core: transaction entry, category breakdown, monthly summary.
• Stretch idea: budget alerts and trend analytics.

Inventory Management System
• Build core: product catalog, stock in/out, low inventory alerts.
• Stretch idea: reorder recommendation engine.

Student Management System
• Build core: student profiles, marks, attendance, reports.
• Stretch idea: parent-friendly progress snapshots.

Task Management CLI
• Build core: task create/update/status/priority workflows.
• Stretch idea: due-date reminders and tag-based filters.

Weather Dashboard
• Build core: city search, current weather, forecast summaries.
• Stretch idea: cached offline report exports.

REST API Client
• Build core: authenticated API calls, response parsing, retries.
• Stretch idea: response caching and pagination helper.

Personal Finance Manager
• Build core: income/expense tracking, savings summary, goals.
• Stretch idea: recommendation insights from spending patterns.`
            },
            {
                title: "Project Deliverables",
                content: `Your final submission must include these deliverable packs.

Architecture Pack:
• clean layered architecture and modular code boundaries
• OOP model design with clear responsibilities

Reliability Pack:
• robust file handling and exception management
• logging and configuration management integrated into runtime

Quality Pack:
• unit tests for core features and failure scenarios
• deterministic behavior under valid and invalid input

Documentation Pack:
• README with setup, run steps, and feature walkthrough
• architecture rationale and trade-off explanation
• short demo guide for interview presentation`
            },
            {
                title: "Final Assessment",
                content: `Purpose: this tab is the final job-readiness checkpoint, not just another quiz.
Why it exists: it validates your ability to build, debug, explain, and review production-grade Python code.

Assessment Format:
• 50 MCQs covering Python fundamentals through production practices
• coding challenges for real implementation scenarios
• debugging exercises with broken workflows
• interview questions and practical assignments
• Python best-practices and code-review checklist evaluation
• course completion summary reflection

Pass Criteria:
• concept clarity and correct reasoning
• working modular implementation quality
• effective debugging and code review communication
• strong project walkthrough confidence`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
• implement one advanced capstone feature (analytics, export, permissions, or optimization)
• maintain architecture boundaries and test coverage while adding feature

Success Check:
• new feature integrates without regressions
• documentation and tests are updated with clear examples`
            },
            {
                title: "Interview Preparation",
                content: `Interview focus:
• architecture decisions and trade-offs in your capstone
• validation and exception flow explanation
• logging/config/test strategy in production context
• concise 5-minute end-to-end demo narrative`
            },
            {
                title: "Module Summary",
                content: `You now have a complete Python capstone path with clear project options, deliverables, and assessment-driven interview readiness.`
            },
            {
                title: "Course Completion Path",
                content: `Complete final assessment, polish capstone documentation, and prepare your project walkthrough for portfolio and interviews.`
            }
        ]
    },
    {
        number: "Module 7",
        title: "Python for Real-World Development (Optional)",
        description: "Choose one specialization track to align Python skills with your career goal: Backend, Automation, or Data & AI Foundations.",
        duration: "75 min",
        lessons: "10 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Backend Development Track", "Automation & Scripting Track", "Data & AI Foundations Track", "Career Track Selection"],
        detailedDescription: "Optional specialization module that personalizes learning toward practical career outcomes after core Python mastery.",
        detailedContent: [
            {
                title: "Module Blueprint",
                content: `Module Number: Module 7
Module Name: Python for Real-World Development (Optional)
Difficulty: Intermediate to Advanced
Estimated Reading Time: 75 min
Estimated Completion Time: 8-10 hours
Prerequisites: Modules 1-6
Learning Objectives:
• choose specialization based on career direction
• gain applied skills in a focused Python domain
• build a track-specific portfolio artifact
Skills Gained:
• focused domain implementation depth
• stronger career positioning
• personalized learning pathway`
            },
            {
                title: "Specialization Tracks",
                content: `Choose one track based on your target role.

Backend Development Track
• Flask
• FastAPI
• REST APIs
• Authentication
• SQLAlchemy
• PostgreSQL

Automation & Scripting Track
• Web Scraping
• Excel Automation
• Email Automation
• PDF Processing
• Scheduling Jobs

Data & AI Foundations Track
• NumPy
• Pandas
• Matplotlib
• Basic Machine Learning
• Jupyter Notebooks`
            },
            {
                title: "Mini Challenge",
                content: `Stretch Goal:
• select one specialization track and implement a focused proof-of-concept
• include README with track rationale and next-learning roadmap

Success Check:
• project clearly demonstrates chosen track outcomes
• implementation aligns with career goal and extends capstone skills`
            },
            {
                title: "Module Summary",
                content: `This optional module makes the course more personalized and career-oriented by letting learners branch into practical specialization tracks.`
            },
            {
                title: "Course Completion Path",
                content: `Complete one specialization artifact and add it to your portfolio alongside the capstone project for stronger job positioning.`
            }
        ]
    }
];

courseData.csharpProgramming = [
    {
        number: "Module 1",
        title: "C# Fundamentals and .NET Basics",
        description: "Understand C# syntax, types, methods, and how C# fits into the .NET ecosystem.",
        duration: "50 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: [".NET and CLR", "Types", "Methods", "Conditionals and loops", "Console apps"],
        detailedDescription: "C# is a modern, strongly typed language widely used for backend systems, enterprise applications, and tooling in the .NET ecosystem.",
        detailedContent: [
            {
                title: "C# and the .NET runtime",
                content: `C# code compiles into Intermediate Language (IL), which runs on the Common Language Runtime (CLR). The runtime handles memory, exceptions, and execution rules.

This gives C# a safe and structured environment for enterprise software.`,
                code: `using System;

class Program
{
    static void Main()
    {
        Console.WriteLine("Hello from C#");
    }
}`
            },
            {
                title: "Types and flow control",
                content: `C# combines strong typing with familiar control structures. Start by mastering variables, loops, branches, and method definitions before moving into architecture.`,
                code: `int age = 25;
string name = "Kiran";

if (age >= 18)
{
    Console.WriteLine(name + " is eligible.");
}`
            },
            {
                title: "Readable method-based code",
                content: `Professional C# code tends to be explicit, well-structured, and easy to maintain. Good method boundaries are one of the earliest habits worth building.`,
                code: `static int Add(int a, int b)
{
    return a + b;
}

Console.WriteLine(Add(10, 15));`
            }
        ]
    },
    {
        number: "Module 2",
        title: "Classes, OOP, and Collections",
        description: "Learn classes, properties, constructors, inheritance, interfaces, and C# collections.",
        duration: "55 min",
        lessons: "5 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Classes and properties", "Constructors", "Interfaces", "Lists and dictionaries", "Encapsulation"],
        detailedDescription: "This module covers the building blocks of maintainable C# applications and the object-oriented patterns common in .NET projects.",
        detailedContent: [
            {
                title: "Properties and constructors",
                content: `C# often exposes data through properties rather than public fields. This keeps access controlled while preserving clean syntax.`,
                code: `class Employee
{
    public string Name { get; set; }
    public int Experience { get; set; }

    public Employee(string name, int experience)
    {
        Name = name;
        Experience = experience;
    }
}`
            },
            {
                title: "Interfaces and abstraction",
                content: `Interfaces define contracts. In real systems, they make substitution, testing, and layered design significantly easier.`,
                code: `interface INotifier
{
    void Send(string message);
}

class EmailNotifier : INotifier
{
    public void Send(string message)
    {
        Console.WriteLine("Email: " + message);
    }
}`
            },
            {
                title: "Collections in C#",
                content: `Generic collections like List<T> and Dictionary<TKey, TValue> provide type-safe data handling and are used constantly in .NET codebases.`,
                code: `var names = new List<string> { "A", "B", "C" };
var scores = new Dictionary<string, int>
{
    ["math"] = 92,
    ["cs"] = 97
};`
            }
        ]
    },
    {
        number: "Module 3",
        title: "LINQ, Exceptions, and Practical C#",
        description: "Use LINQ effectively, handle exceptions, and write cleaner everyday .NET code.",
        duration: "55 min",
        lessons: "5 lessons",
        isNew: false,
        isLocked: false,
        topics: ["LINQ basics", "Filtering and projection", "Exception handling", "Using statements", "Clean coding"],
        detailedDescription: "This module covers the parts of C# that make professional code more expressive and maintainable, especially LINQ and structured exception handling.",
        detailedContent: [
            {
                title: "LINQ",
                content: `LINQ lets you query collections declaratively. It improves readability when used with discipline and can replace a lot of repetitive loop-based code.`,
                code: `var numbers = new List<int> { 1, 2, 3, 4, 5 };
var evenSquares = numbers
    .Where(n => n % 2 == 0)
    .Select(n => n * n)
    .ToList();`
            },
            {
                title: "Exceptions",
                content: `C# exception handling should be used deliberately. A professional developer knows when to catch, when to wrap context, and when to fail fast.`,
                code: `try
{
    int value = int.Parse("abc");
}
catch (FormatException)
{
    Console.WriteLine("Invalid number format");
}`
            },
            {
                title: "Everyday code quality",
                content: `Professional C# code is readable, testable, and explicit where it needs to be. That usually matters more than using every language feature available.`,
                code: `string FormatUser(string firstName, string lastName)
{
    return $"{firstName} {lastName}".Trim();
}`
            }
        ]
    },
    {
        number: "Module 4",
        title: "Async Programming and Modern C#",
        description: "Understand async/await, tasks, and the patterns used in scalable and responsive C# applications.",
        duration: "60 min",
        lessons: "5 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Task", "async/await", "I/O-bound work", "Error flow", "Modern language habits"],
        detailedDescription: "Modern C# relies heavily on asynchronous programming, especially in APIs and cloud applications. This module helps you reason correctly about async work.",
        detailedContent: [
            {
                title: "Why async matters",
                content: `Async improves responsiveness and throughput for I/O-bound work such as API calls and file operations. It is not about magically speeding up every task.`,
                code: `async Task<string> GetMessageAsync()
{
    await Task.Delay(500);
    return "done";
}`
            },
            {
                title: "The async/await model",
                content: `The important idea is that an async method may pause, free the thread, and resume later. Once you understand that flow, async code becomes far less mysterious.`,
                code: `static async Task Main()
{
    string message = await GetMessageAsync();
    Console.WriteLine(message);
}`
            },
            {
                title: "Professional async habits",
                content: `Good async code avoids blocking calls like .Result, propagates cancellation when possible, and keeps naming and exception handling consistent.`,
                code: `using var cts = new CancellationTokenSource();
cts.CancelAfter(TimeSpan.FromSeconds(2));`
            }
        ]
    },
    {
        number: "Module 5",
        title: "Web APIs, Architecture, and Professional C#",
        description: "Connect C# fundamentals to real backend development with APIs, layering, and maintainable architecture.",
        duration: "65 min",
        lessons: "5 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Controllers and services", "DTOs", "Dependency injection", "Validation", "Clean architecture"],
        detailedDescription: "This module shows how C# is used in real applications: API layers, service boundaries, validation, and maintainable project structure.",
        detailedContent: [
            {
                title: "Backend structure",
                content: `Production C# projects often separate concerns into controllers, services, repositories, DTOs, and domain models.

That structure supports maintainability and clearer testing boundaries.`,
                code: `public class UserService
{
    public string GetUserName(int id)
    {
        return "Demo User";
    }
}`
            },
            {
                title: "Dependency injection",
                content: `Dependency injection reduces coupling by giving classes what they need instead of forcing them to create every dependency themselves.`,
                code: `public class UserController
{
    private readonly UserService _service;

    public UserController(UserService service)
    {
        _service = service;
    }
}`
            },
            {
                title: "Professional .NET mindset",
                content: `A professional C# engineer can write readable code, reason about async and exceptions, structure services clearly, and build software that teammates can extend safely.`,
                code: `record UserDto(int Id, string Name);

var dto = new UserDto(1, "Belagam Harini");
Console.WriteLine(dto.Name);`
            }
        ]
    },
    {
        number: "Module 6",
        title: "C# Capstone Mini Project",
        description: "Build a clean C# Web API mini project with DTOs, validation, and layered architecture.",
        duration: "70 min",
        lessons: "5 lessons",
        isNew: false,
        isLocked: false,
        topics: ["API contract", "Services", "Validation", "DI setup", "Portfolio output"],
        detailedDescription: "Turn C# fundamentals into a backend-style capstone demonstrating maintainable engineering practices.",
        detailedContent: [
            {
                title: "Project brief",
                content: `Build a Task Tracker API with create/list/update endpoints and status transitions.`
            },
            {
                title: "Implementation expectations",
                content: `Required structure:
• DTOs for request/response
• service layer for core logic
• validation for user inputs
• clear error responses`
            },
            {
                title: "Capstone deliverables",
                content: `Final output should include:
• runnable API project
• dependency-injected services
• unit tests for business logic
• README with endpoint docs and sample payloads`,
                code: `public record TaskItemDto(int Id, string Title, string Status);

public interface ITaskService
{
    TaskItemDto Create(string title);
}`
            }
        ]
    }
];

courseData.cppProgramming = [
    {
        number: "Module 1",
        title: "C++ Syntax and Core Fundamentals",
        description: "Learn C++ syntax, variables, functions, control flow, and compilation basics.",
        duration: "55 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Compilation", "Types", "Functions", "Control flow", "Input/output"],
        detailedDescription: "This module introduces C++ as a compiled systems language and explains how to write basic programs with precision and control.",
        detailedContent: [
            {
                title: "How C++ programs run",
                content: `C++ is compiled directly to machine code. That gives it strong performance and fine-grained control over memory and execution.`,
                code: `#include <iostream>
using namespace std;

int main() {
    cout << "Hello, C++" << endl;
    return 0;
}`
            },
            {
                title: "Variables, flow, and functions",
                content: `C++ gives you direct control with familiar procedural tools: variables, functions, conditionals, and loops. Precision matters more here than in many higher-level languages.`,
                code: `int add(int a, int b) {
    return a + b;
}

int main() {
    int score = 88;
    if (score > 75) {
        cout << "Good job" << endl;
    }
}`
            },
            {
                title: "Why C++ still matters",
                content: `C++ teaches memory awareness, cost models, ownership, and abstraction boundaries. Even if you later use higher-level languages, it sharpens core engineering instincts.`,
                code: `double price = 199.99;
char grade = 'A';
bool ready = true;

cout << price << " " << grade << " " << ready << endl;`
            }
        ]
    },
    {
        number: "Module 2",
        title: "Pointers, References, and Memory",
        description: "Understand memory addresses, references, pointers, stack vs heap, and common C++ pitfalls.",
        duration: "60 min",
        lessons: "5 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Pointers", "References", "Stack and heap", "Dynamic allocation", "Memory safety"],
        detailedDescription: "This is where C++ becomes truly different from many high-level languages: you work closer to memory and need stronger ownership discipline.",
        detailedContent: [
            {
                title: "References vs pointers",
                content: `A reference is an alias; a pointer stores an address. Understanding when each is appropriate is central to writing safe C++ interfaces.`,
                code: `int value = 10;
int& ref = value;
int* ptr = &value;

cout << ref << endl;
cout << *ptr << endl;`
            },
            {
                title: "Stack, heap, and ownership",
                content: `Stack memory is automatic. Heap memory is flexible, but it creates ownership complexity. Senior C++ work means reasoning clearly about who owns what and for how long.`,
                code: `int* data = new int(42);
cout << *data << endl;
delete data;`
            },
            {
                title: "Why memory bugs matter",
                content: `Leaks, dangling pointers, and invalid access are not academic issues. They break production systems and can create serious security risks.`,
                code: `#include <memory>

auto value = std::make_unique<int>(99);
cout << *value << endl;`
            }
        ]
    },
    {
        number: "Module 3",
        title: "STL and Object-Oriented C++",
        description: "Use STL containers, algorithms, classes, and constructors effectively.",
        duration: "60 min",
        lessons: "5 lessons",
        isNew: false,
        isLocked: false,
        topics: ["vector and map", "Algorithms", "Classes", "Constructors", "Encapsulation"],
        detailedDescription: "This module shows how productive C++ can be when you combine STL containers with clear object modeling and structured design.",
        detailedContent: [
            {
                title: "STL containers and algorithms",
                content: `The Standard Template Library gives C++ much of its practical power. Learn vectors, maps, and algorithms before reinventing basic structures.`,
                code: `#include <algorithm>
#include <vector>

vector<int> values = {4, 1, 3, 2};
sort(values.begin(), values.end());`
            },
            {
                title: "Classes and constructors",
                content: `C++ supports object-oriented design, but unlike Java or C#, the language also demands stronger attention to lifetime and ownership.`,
                code: `class Student {
public:
    string name;
    int marks;

    Student(string n, int m) : name(n), marks(m) {}
};`
            },
            {
                title: "Readable abstraction in C++",
                content: `The best C++ code uses abstraction carefully. It keeps performance-aware design without sacrificing readability or safety.`,
                code: `vector<string> languages = {"C++", "Java", "Python"};
for (const auto& item : languages) {
    cout << item << endl;
}`
            }
        ]
    },
    {
        number: "Module 4",
        title: "Modern C++: RAII, Smart Pointers, and Templates",
        description: "Learn the safer and more expressive side of modern C++ with RAII, templates, and smart pointers.",
        duration: "65 min",
        lessons: "5 lessons",
        isNew: false,
        isLocked: false,
        topics: ["RAII", "unique_ptr", "shared_ptr", "Templates", "Generic design"],
        detailedDescription: "Modern C++ avoids many classic memory problems through ownership-oriented design. This module introduces the habits that matter in professional code.",
        detailedContent: [
            {
                title: "RAII",
                content: `RAII means resources are tied to object lifetime. This is one of the most important ideas in C++, because it turns cleanup into a structural guarantee instead of a manual hope.`,
                code: `{
    std::vector<int> data = {1, 2, 3};
} // automatic cleanup here`
            },
            {
                title: "Smart pointers",
                content: `Smart pointers express ownership directly. unique_ptr means one owner; shared_ptr means shared ownership with overhead and care.`,
                code: `#include <memory>

auto user = std::make_unique<int>(42);
std::cout << *user << std::endl;`
            },
            {
                title: "Templates",
                content: `Templates let you write generic, reusable, high-performance code. They are central to the STL and to many serious C++ libraries.`,
                code: `template <typename T>
T maxValue(T a, T b) {
    return a > b ? a : b;
}`
            }
        ]
    },
    {
        number: "Module 5",
        title: "Performance, Concurrency, and Professional C++",
        description: "Connect C++ fundamentals to real systems work with performance thinking, concurrency basics, and maintainable engineering habits.",
        duration: "70 min",
        lessons: "5 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Cost models", "Move semantics", "Threads", "Profiling mindset", "Systems discipline"],
        detailedDescription: "This final module helps learners think like professional C++ engineers: measure cost, manage resources, and reason carefully about concurrency.",
        detailedContent: [
            {
                title: "Thinking in cost models",
                content: `C++ gives you performance, but only if you understand where the cost actually comes from: allocations, copies, cache patterns, and indirection.`,
                code: `std::vector<int> values;
values.reserve(1000); // reduce reallocations`
            },
            {
                title: "Concurrency basics",
                content: `C++ threads are powerful, but shared mutable state is risky. Professional code keeps synchronization explicit and ownership clear.`,
                code: `#include <thread>

void runTask() {
    std::cout << "Task running" << std::endl;
}

std::thread worker(runTask);
worker.join();`
            },
            {
                title: "What makes a C++ developer feel pro",
                content: `A professional C++ engineer can write readable code, manage ownership correctly, use STL effectively, and optimize based on evidence rather than guesswork.`,
                code: `auto values = std::vector<int>{1, 2, 3, 4};
for (const auto& value : values) {
    std::cout << value << std::endl;
}`
            }
        ]
    },
    {
        number: "Module 6",
        title: "C++ Capstone Mini Project",
        description: "Build a performance-aware C++ mini project with STL design, ownership safety, and measurable output.",
        duration: "75 min",
        lessons: "5 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Problem scoping", "Data structures", "Ownership model", "Benchmarking", "Portfolio output"],
        detailedDescription: "Apply modern C++ patterns in a compact project that demonstrates engineering discipline and systems thinking.",
        detailedContent: [
            {
                title: "Project brief",
                content: `Build a Log Analyzer that reads structured logs, aggregates metrics, and reports top error categories.`
            },
            {
                title: "Engineering focus",
                content: `Your implementation should emphasize:
• STL-first data structures
• safe ownership (value semantics / smart pointers)
• deterministic resource handling
• basic timing benchmarks`
            },
            {
                title: "Capstone deliverables",
                content: `Portfolio-ready output:
• compiled CLI with sample input
• metrics summary report
• one measured optimization improvement
• README describing complexity and trade-offs`,
                code: `std::unordered_map<std::string, int> errorCount;
for (const auto& line : lines) {
    if (line.find("ERROR") != std::string::npos) {
        errorCount[extractCode(line)]++;
    }
}`
            }
        ]
    }
];
