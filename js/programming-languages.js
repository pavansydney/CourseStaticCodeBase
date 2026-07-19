// ============================================================
// Programming Languages track content.
// Loaded only on the Courses page (after script.js). It extends the
// existing global `courseData` object with language-specific tracks.
// ============================================================

/* global courseData */

courseData.javaProgramming = [
    {
        number: "Module 1",
        title: "Java Basics and Syntax",
        description: "Learn Java syntax, types, control flow, methods, and how Java programs run on the JVM.",
        duration: "50 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["JDK and JVM", "Variables and types", "Control flow", "Methods", "Compilation and execution"],
        detailedDescription: "This module gives you a solid Java foundation: source code, compilation, runtime behavior, and clean method-based program structure.",
        detailedContent: [
            {
                title: "How Java code runs",
                content: `Java source code is compiled into <strong>bytecode</strong>, which is then executed by the <strong>Java Virtual Machine (JVM)</strong>. This design gives Java portability and strong runtime tooling.

Core flow:
• write a .java file
• compile with javac
• run the class with java`,
                code: `public class HelloJava {
    public static void main(String[] args) {
        System.out.println("Hello, Java");
    }
}`
            },
            {
                title: "Types and variables",
                content: `Java is strongly and statically typed. Primitive types model simple values, while reference types represent objects, arrays, and strings.

The benefit is clarity and strong compiler validation, especially in larger codebases.`,
                code: `int age = 24;
double score = 91.5;
boolean ready = true;
String name = "Harini";`
            },
            {
                title: "Conditions, loops, and methods",
                content: `Methods are the first major step toward writing structured code. Pair them with if/else and loops so logic is reusable instead of repeated.

Small methods with clear names make Java code easier to read and maintain.`,
                code: `static int add(int a, int b) {
    return a + b;
}

for (int i = 1; i <= 3; i++) {
    System.out.println(add(i, 10));
}`
            }
        ]
    },
    {
        number: "Module 2",
        title: "Object-Oriented Programming in Java",
        description: "Model data and behavior with classes, constructors, encapsulation, inheritance, and interfaces.",
        duration: "55 min",
        lessons: "5 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Classes and objects", "Constructors", "Encapsulation", "Inheritance", "Interfaces"],
        detailedDescription: "This module covers the object-oriented style that makes Java such a strong language for enterprise systems and backend applications.",
        detailedContent: [
            {
                title: "Classes and objects",
                content: `A class describes a type of thing; an object is a specific instance. Constructors make sure objects start in a valid state.

Good object design begins by asking: what data belongs together, and what behavior should live with that data?`,
                code: `class Student {
    String name;
    int marks;

    Student(String name, int marks) {
        this.name = name;
        this.marks = marks;
    }
}`
            },
            {
                title: "Encapsulation",
                content: `Encapsulation protects internal state. Use private fields and expose controlled behavior through methods or getters when appropriate.

This reduces invalid state changes and keeps responsibilities clear.`,
                code: `class BankAccount {
    private double balance;

    void deposit(double amount) {
        if (amount > 0) balance += amount;
    }

    double getBalance() {
        return balance;
    }
}`
            },
            {
                title: "Inheritance and interfaces",
                content: `Inheritance is one way to reuse behavior, but Java developers often prefer interfaces and composition for flexibility.

Interfaces define contracts. That makes systems easier to extend and test.`,
                code: `interface Payment {
    void pay(double amount);
}

class CardPayment implements Payment {
    public void pay(double amount) {
        System.out.println("Paid " + amount);
    }
}`
            }
        ]
    },
    {
        number: "Module 3",
        title: "Collections, Generics, and Exceptions",
        description: "Work with Java collections, generics, and the exception model that supports robust code.",
        duration: "55 min",
        lessons: "5 lessons",
        isNew: false,
        isLocked: false,
        topics: ["List and Map", "Set", "Generics", "Checked exceptions", "Exception handling"],
        detailedDescription: "This module covers the daily Java tools used in backend, enterprise, and production codebases.",
        detailedContent: [
            {
                title: "Collections framework",
                content: `The Java Collections Framework gives you reusable data structures with well-understood performance characteristics.

Learn when to choose List, Set, and Map based on order, uniqueness, and lookup needs.`,
                code: `List<String> skills = new ArrayList<>();
skills.add("Java");

Map<String, Integer> scores = new HashMap<>();
scores.put("OOP", 95);`
            },
            {
                title: "Generics",
                content: `Generics let Java express reusable code without giving up type safety. They eliminate many unsafe casts and make APIs clearer.

This is one of the reasons Java collections stay readable at scale.`,
                code: `List<Integer> numbers = new ArrayList<>();
numbers.add(10);
numbers.add(20);

for (Integer value : numbers) {
    System.out.println(value * 2);
}`
            },
            {
                title: "Exceptions",
                content: `Java exceptions model failures explicitly. A professional developer should know when to recover, when to wrap, and when to let a failure propagate.

Reliable systems come from deliberate failure handling, not silence.`,
                code: `try {
    int result = 10 / 0;
    System.out.println(result);
} catch (ArithmeticException ex) {
    System.out.println("Cannot divide by zero");
}`
            }
        ]
    },
    {
        number: "Module 4",
        title: "Streams, Lambdas, and Modern Java",
        description: "Use lambdas, streams, and Optional to write cleaner modern Java code.",
        duration: "60 min",
        lessons: "5 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Lambdas", "Stream pipeline", "map/filter/reduce", "Optional", "Modern style"],
        detailedDescription: "Modern Java is more expressive than the old loop-heavy style. This module shows how to process data clearly and safely.",
        detailedContent: [
            {
                title: "Lambdas and functional style",
                content: `Lambdas made Java far more concise by letting developers pass behavior directly.

They matter not just because they save lines, but because they unlock cleaner abstractions around data processing.`,
                code: `List<String> names = Arrays.asList("Asha", "Ravi", "Mohan");
names.forEach(name -> System.out.println(name));`
            },
            {
                title: "Streams",
                content: `Streams work best when each transformation has one clear purpose: filter, map, sort, reduce, or collect.

The important skill is writing pipelines that stay readable rather than turning into one long chain of mystery.`,
                code: `List<Integer> scores = Arrays.asList(91, 74, 88, 96);
List<Integer> topScores = scores.stream()
    .filter(score -> score >= 90)
    .collect(Collectors.toList());`
            },
            {
                title: "Optional and safer APIs",
                content: `Optional communicates absence explicitly. Used well, it makes APIs safer. Used everywhere blindly, it becomes noise.

The mature habit is to use it where it improves clarity, especially in return types.`,
                code: `Optional<String> email = Optional.ofNullable(findEmail(userId));
email.ifPresent(value -> System.out.println(value));`
            }
        ]
    },
    {
        number: "Module 5",
        title: "Concurrency and Backend-Ready Java",
        description: "Understand threads, executors, immutability, and the habits needed for backend Java work.",
        duration: "65 min",
        lessons: "5 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Threads", "Executors", "Immutability", "Race conditions", "Backend mindset"],
        detailedDescription: "This module connects Java syntax to professional backend engineering, where concurrency, state safety, and maintainability matter every day.",
        detailedContent: [
            {
                title: "Threads and tasks",
                content: `Java supports direct thread creation, but production systems usually use executor abstractions to manage concurrency more cleanly.

The key skill is not just starting threads, but understanding how task execution is coordinated safely.`,
                code: `ExecutorService executor = Executors.newFixedThreadPool(2);
executor.submit(() -> System.out.println("Running task"));
executor.shutdown();`
            },
            {
                title: "Immutability and shared state",
                content: `Most concurrency bugs come from shared mutable state. Immutability dramatically reduces those risks.

A backend-ready Java engineer thinks carefully about ownership and data flow across threads.`,
                code: `final class UserProfile {
    private final String name;

    UserProfile(String name) {
        this.name = name;
    }

    String getName() {
        return name;
    }
}`
            },
            {
                title: "Professional backend habits",
                content: `To feel professional in Java, you should be able to:
• structure code into coherent packages and layers
• reason about exceptions and concurrency
• choose simple abstractions before clever ones
• write code that another engineer can safely extend`,
                code: `public record ApiResult(boolean success, String message) {}

ApiResult result = new ApiResult(true, "Saved successfully");
System.out.println(result.message());`
            }
        ]
    },
    {
        number: "Module 6",
        title: "Java Capstone Mini Project",
        description: "Build a mini backend-style Java project with layered design, validation, and testable structure.",
        duration: "70 min",
        lessons: "5 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Project scoping", "Layered packages", "Validation", "Error handling", "Portfolio output"],
        detailedDescription: "Apply Java fundamentals in a practical capstone that mirrors real backend workflow and code organization.",
        detailedContent: [
            {
                title: "Project brief",
                content: `Build a Student Progress Tracker with modules, scores, and completion flags. Keep it structured and easy to extend.`
            },
            {
                title: "Suggested architecture",
                content: `Use a simple layered design:
• model (Student, ModuleProgress)
• service (ProgressService)
• app (main flow and user interaction)

Document choices in a short README for portfolio clarity.`
            },
            {
                title: "Capstone deliverables",
                content: `By the end, learners should ship:
• runnable project with clean package structure
• input validation + exception-safe flows
• at least 3 test scenarios
• README with architecture notes and trade-offs`,
                code: `public class ProgressService {
    public double completionRate(int done, int total) {
        if (total <= 0) throw new IllegalArgumentException("total must be > 0");
        return (done * 100.0) / total;
    }
}`
            }
        ]
    }
];

courseData.pythonProgramming = [
    {
        number: "Module 1",
        title: "Python Basics and Flow Control",
        description: "Learn Python syntax, variables, conditions, loops, and functions with beginner-friendly examples.",
        duration: "45 min",
        lessons: "5 lessons",
        isNew: true,
        isLocked: false,
        topics: ["Python runtime", "Variables", "if and loops", "Functions", "Input and output"],
        detailedDescription: "This module gives you the Python foundation needed for scripting, automation, data work, and backend development.",
        detailedContent: [
            {
                title: "Why Python feels different",
                content: `Python is known for readable syntax and fast iteration speed. It is interpreted, expressive, and productive for both beginners and professionals.

The same language can power quick scripts, machine learning workflows, and web services.`,
                code: `name = "Harini"
age = 23

print(f"{name} is {age} years old")`
            },
            {
                title: "Indentation and flow control",
                content: `Python uses indentation as part of the syntax. That makes structure visible by default, but it also means formatting mistakes are logic mistakes.

Good Python is readable first, clever second.`,
                code: `score = 82

if score >= 75:
    print("Passed with distinction")
else:
    print("Keep practicing")`
            },
            {
                title: "Functions and reuse",
                content: `Functions are central to Python's readability. Short, well-named functions make scripts easier to test, debug, and grow into real applications.`,
                code: `def greet(name):
    return f"Hello, {name}!"

print(greet("Python learner"))`
            }
        ]
    },
    {
        number: "Module 2",
        title: "Python Data Structures",
        description: "Use lists, tuples, sets, dictionaries, slicing, and comprehensions effectively.",
        duration: "50 min",
        lessons: "5 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Lists", "Tuples", "Sets", "Dictionaries", "Comprehensions"],
        detailedDescription: "Python becomes much more powerful once you understand its built-in data structures and how to transform data cleanly.",
        detailedContent: [
            {
                title: "Lists and dictionaries",
                content: `Lists and dictionaries dominate everyday Python. They are flexible, readable, and powerful enough for most application-level work.`,
                code: `skills = ["python", "sql", "ml"]
profile = {"name": "Asha", "experience": 2}

print(skills[0])
print(profile["name"])`
            },
            {
                title: "Sets, tuples, and mutability",
                content: `Use tuples for fixed records, sets for uniqueness, and lists when order and mutation matter.

Learning mutability early prevents many subtle bugs later.`,
                code: `coords = (10, 20)
visited = {"api", "auth", "db"}

if "api" in visited:
    print("API already visited")`
            },
            {
                title: "Comprehensions and readable transformations",
                content: `Comprehensions are one of Python's signature strengths. They let you transform data compactly, but readability is still the standard.`,
                code: `numbers = [1, 2, 3, 4, 5]
squares = [n * n for n in numbers if n % 2 == 1]
print(squares)`
            }
        ]
    },
    {
        number: "Module 3",
        title: "Files, Errors, and Practical Scripting",
        description: "Read files, handle exceptions, work with modules, and write practical Python scripts.",
        duration: "50 min",
        lessons: "5 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Reading files", "Writing files", "Exceptions", "Modules", "CLI scripting"],
        detailedDescription: "This module moves Python from beginner syntax into useful automation: reading data, handling failures, and structuring scripts cleanly.",
        detailedContent: [
            {
                title: "File handling",
                content: `File I/O is central to automation, backend scripting, and data pipelines. The with-statement is the standard pattern because it handles cleanup safely.`,
                code: `with open("notes.txt", "w", encoding="utf-8") as file:
    file.write("Practice Python every day")`
            },
            {
                title: "Exceptions",
                content: `Exceptions are how Python reports runtime problems. A professional developer knows when to catch, when to re-raise, and when to fail fast.`,
                code: `try:
    age = int("twenty")
except ValueError:
    print("Please enter a valid integer")`
            },
            {
                title: "Practical scripting mindset",
                content: `A useful script should validate inputs, produce clear outputs, and be easy to rerun.

The syntax matters, but the workflow mindset matters more.`,
                code: `def summarize_scores(scores):
    return sum(scores) / len(scores)

print(summarize_scores([80, 90, 100]))`
            }
        ]
    },
    {
        number: "Module 4",
        title: "Python OOP, Modules, and Environments",
        description: "Learn classes, modules, packages, virtual environments, and clean project organization in Python.",
        duration: "55 min",
        lessons: "5 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Classes", "Modules", "Packages", "venv", "Project structure"],
        detailedDescription: "Python is more than scripts. This module teaches the structure you need for maintainable codebases: object modeling, reusable modules, and environment isolation.",
        detailedContent: [
            {
                title: "Classes and object modeling",
                content: `Python supports OOP, but good Python uses classes only when they make a model clearer.

The professional question is always: does this class clarify the system, or is a function enough?`,
                code: `class Student:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

    def passed(self):
        return self.marks >= 40`
            },
            {
                title: "Modules and packages",
                content: `As projects grow, code must be split into modules and packages. That keeps responsibilities clear and prevents giant script files from becoming unmaintainable.`,
                code: `# utils.py
def format_name(first, last):
    return f"{first} {last}".strip()`
            },
            {
                title: "Virtual environments and dependency discipline",
                content: `Professional Python work uses isolated environments so dependencies remain reproducible across machines and projects.

That is the baseline for reliable collaboration.`,
                code: `python -m venv .venv
.venv\\Scripts\\activate
pip install requests pytest`
            }
        ]
    },
    {
        number: "Module 5",
        title: "APIs, Testing, and Production Python",
        description: "Write testable Python, call APIs, and adopt the habits needed for production-grade code.",
        duration: "60 min",
        lessons: "5 lessons",
        isNew: false,
        isLocked: false,
        topics: ["HTTP APIs", "JSON", "Unit testing", "Logging", "Production habits"],
        detailedDescription: "This module turns Python knowledge into professional practice: working with APIs, testing behavior, and writing code that is safe to maintain and deploy.",
        detailedContent: [
            {
                title: "Calling APIs",
                content: `A large amount of practical Python work involves HTTP APIs. You need to handle response codes, parse JSON, and defend against unreliable networks.`,
                code: `import requests

response = requests.get("https://api.github.com")
print(response.status_code)
print(response.json())`
            },
            {
                title: "Testing with confidence",
                content: `Tests protect behavior and make refactoring safer. Even simple unit tests add a lot of value in Python because the language is dynamic.`,
                code: `def add(a, b):
    return a + b

def test_add():
    assert add(2, 3) == 5`
            },
            {
                title: "Production habits",
                content: `To feel professional in Python, you should be able to structure projects, validate assumptions, test core behavior, and log failures clearly.

That is what separates scripts from maintainable systems.`,
                code: `import logging

logging.basicConfig(level=logging.INFO)
logging.info("Application started")`
            }
        ]
    },
    {
        number: "Module 6",
        title: "Python Capstone Mini Project",
        description: "Build a practical Python automation project with testing, error handling, and clean structure.",
        duration: "65 min",
        lessons: "5 lessons",
        isNew: false,
        isLocked: false,
        topics: ["Project definition", "CLI flow", "Validation", "Pytest coverage", "Portfolio output"],
        detailedDescription: "Create a project that feels production-like: clear inputs, robust processing, and documented behavior.",
        detailedContent: [
            {
                title: "Project brief",
                content: `Build an Expense Summary CLI that reads transactions from CSV, validates records, and outputs monthly summaries.`
            },
            {
                title: "Quality targets",
                content: `Minimum quality bar:
• handle malformed rows safely
• validate categories and dates
• separate parsing and business logic
• log key failures cleanly`
            },
            {
                title: "Capstone deliverables",
                content: `Publish:
• CLI entrypoint with help usage
• tested parser + summary logic
• at least 5 unit tests
• README with assumptions and sample output`,
                code: `def summarize_expenses(records):
    totals = {}
    for item in records:
        month = item["date"][:7]
        totals[month] = totals.get(month, 0.0) + item["amount"]
    return totals`
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
