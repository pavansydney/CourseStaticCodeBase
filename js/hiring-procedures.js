(function () {
    var companyData = [
        {
            name: "Microsoft",
            group: "product",
            tag: "Cloud + Product Engineering",
            lastUpdated: "Preparation example; not independently verified",
            aliases: ["msft", "azure"],
            process: [
                "Application + recruiter screening",
                "Online coding assessment or shortlisting by profile",
                "2-4 technical rounds (DSA + design + problem solving)",
                "Managerial/behavioral loop",
                "Offer and team matching"
            ],
            rounds: [
                { name: "Coding", check: "Problem solving, clean coding, test cases", clear: "Think aloud, optimize progressively, discuss complexity clearly." },
                { name: "System Design", check: "Scalability, APIs, trade-offs", clear: "Start from requirements, estimate load, defend choices." },
                { name: "Behavioral", check: "Ownership, growth mindset, collaboration", clear: "Use STAR with measurable outcomes and customer impact." }
            ],
            application: {
                fresher: [
                    "Keep resume one page with 2-3 strong projects and measurable outcomes.",
                    "Get referrals from alumni or hackathon contacts before applying.",
                    "Practice OA on LeetCode medium arrays/graphs and timed mocks."
                ],
                experienced: [
                    "Resume should highlight impact metrics, architecture choices, and cross-team execution.",
                    "Tailor profile to role family: platform, backend, AI, cloud, or product engineering.",
                    "Prepare project deep dives with failure recovery and production incident stories."
                ]
            },
            guides: {
                fresher: {
                    focus: [
                        "Master 8-10 high-frequency DSA patterns and dry-run discipline.",
                        "Build one deployable project with testing + README + architecture notes.",
                        "Practice behavioral stories around initiative, teamwork, and learning speed."
                    ],
                    day30: [
                        "Days 1-10: use [DSA + design], Foundations modules 1-2; implement pair indices and prefix-sum counting. Submit empty, duplicate, and negative-input tests plus time/space notes.",
                        "Days 11-20: use [OS + networks] to trace a browser request through DNS, TCP, and HTTP; build a small notes API and document one failing request and its fix.",
                        "Days 21-30: run two 40-minute coding mocks and a five-minute project walkthrough; log one correctness and one communication repair after each."
                    ],
                    day60: [
                        "Days 31-40: study requirement framing in [DSA + design]; estimate a notes service for 100000 users and compare a single database with a read replica.",
                        "Days 41-50: add API tests and a README request/response example; prepare two STAR stories about a teammate disagreement and learning an unfamiliar tool.",
                        "Days 51-60: run two coding/project/behavioral mock loops; re-solve every failed test without hints after three days and compare results with the target role description."
                    ]
                },
                experienced: {
                    focus: [
                        "Strengthen LLD/HLD communication and trade-off reasoning.",
                        "Be ready to discuss production architecture and reliability decisions.",
                        "Demonstrate leadership signals even as IC: mentorship, influence, ownership."
                    ],
                    day30: [
                        "Days 1-10: map the actual engineering posting to three evidence-backed accomplishments; refresh two graph and two DP problems in [DSA + design] with boundary tests.",
                        "Days 11-20: write a two-page service design with traffic, storage, SLOs, and failure assumptions; compare primary reads against lagging replicas.",
                        "Days 21-30: prepare four real stories on incident ownership, mentoring, disagreement, and a failed decision; distinguish your actions from team outcomes."
                    ],
                    day60: [
                        "Days 31-40: use reliability and rollout modules in [DSA + design] to design a canary and rollback for your service; specify two measurable stop gates.",
                        "Days 41-50: run two 45-minute design mocks and two coding screens; revise one weak decision from each using explicit costs and failure behavior.",
                        "Days 51-60: rehearse a project deep dive with probing on scale, access control, and recovery; ask the recruiter to confirm role scope and interview format rather than assuming a fixed loop."
                    ]
                }
            },
            pitfalls: {
                fresher: ["Memorized solutions without explanation", "Weak project depth", "No edge-case testing"],
                experienced: ["Too much theory, little impact evidence", "Weak trade-off discussion", "Generic leadership answers"]
            }
        },
        {
            name: "Amazon",
            group: "product",
            tag: "High-Bar Ownership Culture",
            lastUpdated: "Preparation example; not independently verified",
            aliases: ["aws"],
            process: [
                "Application + recruiter connect",
                "Online assessment (coding + work simulation in some roles)",
                "Technical phone/video rounds",
                "Onsite/virtual loop with strong leadership-principle evaluation",
                "Offer + compensation discussion"
            ],
            rounds: [
                { name: "OA", check: "Coding under time pressure", clear: "Prioritize accuracy first 20 min, then optimize." },
                { name: "Technical", check: "DSA, design, debugging", clear: "State assumptions, validate edge cases, discuss alternatives." },
                { name: "LP Behavioral", check: "Leadership principles alignment", clear: "Use specific situations with measurable outcomes." }
            ],
            application: {
                fresher: [
                    "Use role-specific resumes: SDE, Support Engineer, Data roles differ in keywords.",
                    "Practice OA stamina with full-length timed sessions.",
                    "Prepare LP-aligned examples even for fresher roles."
                ],
                experienced: [
                    "Map every major project to relevant leadership principles before interviews.",
                    "Quantify ownership scope: services, users, revenue impact, reliability metrics.",
                    "Practice bar-raiser style deep probing on decisions and failures."
                ]
            },
            guides: {
                fresher: {
                    focus: ["DSA speed + correctness", "LP story basics", "CS fundamentals for follow-ups"],
                    day30: [
                        "Days 1-10: use [DSA + design] for arrays, hashing, and queue traversal; solve six scoped exercises, recording counterexamples rather than memorized code.",
                        "Days 11-20: take two self-timed 60-minute OA practice sets of two problems; leave ten minutes for tests and classify failures by modeling, coding, or time management.",
                        "Days 21-30: map three truthful project/club stories to ownership, learning, and customer impact; practice STAR and a follow-up about what you personally changed."
                    ],
                    day60: [
                        "Days 31-40: sketch a retry-safe order API using [DSA + design]; explain what happens when the same request is submitted twice.",
                        "Days 41-50: rehearse two behavioral mocks with failure/conflict follow-ups; replace unsupported impact claims with actual evidence or an honest qualitative outcome.",
                        "Days 51-60: run two mixed coding/project/behavioral loops and re-solve missed cases after three days; confirm any OA and LP emphasis in the actual invitation."
                    ]
                },
                experienced: {
                    focus: ["Design depth and scale decisions", "LP depth with conflicts/trade-offs", "Production troubleshooting narratives"],
                    day30: [
                        "Days 1-10: complete four coding refresh problems in [DSA + design], including a directed-cycle and unreachable-DP case; explain correctness aloud.",
                        "Days 11-20: write an order/notification design with 100 requests/s, a 24-hour idempotency window, and worker failure handling; compare outbox with direct publish.",
                        "Days 21-30: prepare five real LP-oriented stories covering ownership, disagreement, failure, customer impact, and learning; record scope and evidence without inventing metrics."
                    ],
                    day60: [
                        "Days 31-40: perform a reliability review of that design: bounded retries, poison jobs, alarms, and an outage recovery trace.",
                        "Days 41-50: run two probing behavioral sessions and two 45-minute design mocks; answer what alternatives you rejected and what you would change.",
                        "Days 51-60: simulate one full coding/design/story loop and repair the weakest two dimensions; clarify level, role responsibilities, and applicable stages with the recruiter."
                    ]
                }
            },
            pitfalls: {
                fresher: ["Ignoring LP rounds", "Rushing OA and losing easy points", "No structured communication"],
                experienced: ["Shallow examples for LP", "Weak metrics", "No failure-postmortem story"]
            }
        },
        {
            name: "Google",
            group: "product",
            tag: "Strong Problem Solving + Structured Thinking",
            lastUpdated: "Preparation example; not independently verified",
            aliases: ["gcp", "alphabet"],
            process: [
                "Application/referral screening",
                "Recruiter call",
                "Technical interview rounds (coding + systems depending level)",
                "Googlyness/behavioral assessment",
                "Hiring committee + team match"
            ],
            rounds: [
                { name: "Coding", check: "Algorithmic depth and clarity", clear: "Communicate brute-force to optimal transition clearly." },
                { name: "Design", check: "Architecture reasoning", clear: "Use requirement-first approach and justify trade-offs." },
                { name: "Googlyness", check: "Collaboration and learning behavior", clear: "Show humility, impact, and ambiguity handling." }
            ],
            application: {
                fresher: ["Prioritize referral-backed applications where possible.", "Create GitHub portfolio with polished README and testing evidence.", "Practice whiteboard-style explanation, not just IDE coding."],
                experienced: ["Bring clear examples of system ownership and scaling outcomes.", "Prepare for deep follow-up questions on decisions and trade-offs.", "Align profile to role ladder expectations."]
            },
            guides: {
                fresher: {
                    focus: ["Core DSA patterns + clean explanation", "Strong fundamentals in OS/DB/Networks", "Project depth with quantifiable outcomes"],
                    day30: [
                        "Days 1-10: work through arrays and traversal in [DSA + design]; write a brute-force and optimized pair-sum solution with distinct-index tests.",
                        "Days 11-20: solve six graph/tree exercises from blank files; draw the visited-state invariant and include disconnected and skewed inputs.",
                        "Days 21-30: record two spoken 40-minute mocks; implement and document one project feature whose code and tests you can explain line by line."
                    ],
                    day60: [
                        "Days 31-40: derive coin-change DP and topological ordering without templates in [DSA + design]; compare outputs with hand-worked tiny cases.",
                        "Days 41-50: write one numeric API design and two STAR stories on ambiguity and collaboration; identify an assumption that changed after feedback.",
                        "Days 51-60: run three mixed mocks with independent problems; reattempt missed categories after three and seven days, and ask which role-specific topics actually apply."
                    ]
                },
                experienced: {
                    focus: ["High signal design communication", "Code quality and correctness under probing", "Cross-team influence examples"],
                    day30: [
                        "Days 1-10: refresh graphs, binary search, and DP in [DSA + design]; solve two per family and explain invariant and worst-case input shape.",
                        "Days 11-20: draft a feed or chat design with numeric load, ordering/freshness requirements, and a stated failure model; compare two partition choices.",
                        "Days 21-30: prepare three architecture narratives distinguishing your decisions, rejected alternatives, measured outcomes, and unknowns."
                    ],
                    day60: [
                        "Days 31-40: use advanced [DSA + design] case studies to probe hot partitions, stale caches, and failed ranking; document one simpler alternative.",
                        "Days 41-50: run two coding and two design mocks plus a collaboration debrief; practice admitting and correcting a mistaken assumption.",
                        "Days 51-60: close two recurring weaknesses with fresh problems and a revised design; prepare questions on team scope and confirm the current selection stages."
                    ]
                }
            },
            pitfalls: {
                fresher: ["Jumping to code without clarification", "Poor communication during solving", "Weak project ownership articulation"],
                experienced: ["Unstructured design answers", "No performance/cost metrics", "Overly generic collaboration stories"]
            }
        },
        {
            name: "Deloitte",
            group: "consulting",
            tag: "Consulting + Delivery + Client Readiness",
            lastUpdated: "Preparation example; not independently verified",
            aliases: ["big4"],
            process: ["Application screening", "Aptitude/technical test", "Technical interview", "Managerial/client-facing discussion", "HR and offer"],
            rounds: [
                { name: "Assessment", check: "Aptitude + role basics", clear: "Time-box sections and avoid spending too long on one question." },
                { name: "Technical", check: "Role stack, debugging, project clarity", clear: "Explain practical implementation and delivery context." },
                { name: "Managerial", check: "Communication, client handling", clear: "Show structured communication and stakeholder awareness." }
            ],
            application: {
                fresher: ["Show internship/project evidence with business use case.", "Prepare aptitude plus role-stack basics (Java/.NET/Cloud/Data).", "Practice communication for client-facing interactions."],
                experienced: ["Highlight delivery ownership, timelines, and client outcomes.", "Prepare scenarios on requirement ambiguity and change handling.", "Demonstrate mentoring and team-coordination ability."]
            },
            guides: {
                fresher: {
                    focus: ["Aptitude speed", "Project explanation clarity", "Professional communication"],
                    day30: [
                        "Days 1-10: inspect the actual job description; choose one language in [Programming languages] and implement an expense-total validator with zero, invalid, and duplicate-record tests.",
                        "Days 11-20: do four 20-minute reasoning/numeracy drills if relevant to the invitation; log accuracy and redo errors rather than assuming a cutoff.",
                        "Days 21-30: explain an expense-tracking project in business terms: user, workflow, your module, tests, and limitation; record two three-minute walkthroughs."
                    ],
                    day60: [
                        "Days 31-40: simulate a client asking to change an expense rule; write three clarifying questions, two acceptance tests, and a revised scope.",
                        "Days 41-50: use [OS + networks] to trace a failed API request; write a short client update stating evidence, next action, and expected follow-up time.",
                        "Days 51-60: run two role-stack/project/communication mocks and one timed test if applicable; revise your weakest explanation and two resume bullets using only real work."
                    ]
                },
                experienced: {
                    focus: ["Consulting-style structured answers", "Delivery metrics and escalation handling", "Leadership in execution"],
                    day30: [
                        "Days 1-10: map three posting requirements to actual delivery evidence; outline one client project with scope, timeline, constraints, and your decisions.",
                        "Days 11-20: use [DSA + design] design-document modules to write an ADR for an integration change, including rejected options and rollout risks.",
                        "Days 21-30: prepare three STAR stories on changing requirements, missed estimates, and escalation; state the trade-off and what you learned."
                    ],
                    day60: [
                        "Days 31-40: simulate a scope increase with a fixed deadline; produce a prioritization table, dependency list, and stakeholder update.",
                        "Days 41-50: run two technical/client mocks with cross-questioning on estimates and risk; rewrite one weak architecture justification.",
                        "Days 51-60: rehearse a ten-minute project review with real quality/timeline evidence; confirm whether the target is consulting, engineering, or another role before final revision."
                    ]
                }
            },
            pitfalls: {
                fresher: ["Ignoring communication quality", "Weak fundamentals", "No business context in project story"],
                experienced: ["Tech-heavy but client-light answers", "No escalation/conflict stories", "Missing timeline/accountability metrics"]
            }
        },
        {
            name: "Accenture",
            group: "consulting",
            tag: "Assessment-Driven + Role Mapping",
            lastUpdated: "Preparation example; not independently verified",
            aliases: ["accenture"],
            process: ["Application", "Cognitive + coding assessments", "Technical interview", "Managerial/HR rounds", "Offer"],
            rounds: [
                { name: "Assessments", check: "Aptitude, communication, coding basics", clear: "Practice full test format and strict timing." },
                { name: "Technical", check: "Project implementation and stack fundamentals", clear: "Use clear architecture and module ownership explanation." },
                { name: "HR", check: "Adaptability and culture fit", clear: "Answer with examples of learning and teamwork." }
            ],
            application: {
                fresher: ["Confirm assessment sections, timing, scoring, and any cutoffs from your actual invitation; formats vary.", "Build one role-relevant data/automation mini-project with a tested input/output example.", "Strengthen spoken explanation and confidence."],
                experienced: ["Map profile to role family and domain vertical.", "Prepare migration, modernization, and delivery stories.", "Show process maturity: quality, risk, and stakeholder communication."]
            },
            guides: {
                fresher: {
                    focus: ["Assessment readiness", "Technical fundamentals", "Communication polish"],
                    day30: [
                        "Days 1-10: select the posting's language in [Programming languages]; build a CSV record-cleaning script with missing-field and duplicate tests.",
                        "Days 11-20: if the invite includes assessments, run three 30-minute mixed reasoning/coding sessions; track errors by section and confirm scoring rules.",
                        "Days 21-30: add a README showing raw input, cleaned output, and your contribution; rehearse a two-minute explanation of learning the chosen tool."
                    ],
                    day60: [
                        "Days 31-40: change the script to support a new field without breaking old input; use [DSA + design] API/schema evolution lessons to describe compatibility.",
                        "Days 41-50: conduct two project and communication mocks; prepare STAR examples on adapting to feedback and coordinating a deadline.",
                        "Days 51-60: repeat a timed set only if role-relevant, then fix two recurring defects; verify stack and assessment format against the actual opportunity."
                    ]
                },
                experienced: {
                    focus: ["Project-to-business impact mapping", "Leadership and collaboration examples", "Domain articulation"],
                    day30: [
                        "Days 1-10: map your resume to one target role/domain; produce a table of three delivery decisions, business impact, and supporting evidence.",
                        "Days 11-20: draft a modernization plan in [DSA + design] with old/new data contracts, migration order, reconciliation, and rollback.",
                        "Days 21-30: prepare four real stories on risk, stakeholder conflict, quality, and learning; practice explaining one to a nontechnical listener."
                    ],
                    day60: [
                        "Days 31-40: simulate a partial migration failure; write a containment plan and a client update distinguishing known facts from estimates.",
                        "Days 41-50: run two stack-depth mocks and one managerial scenario; repair the least defensible technical decision with a small test or trace.",
                        "Days 51-60: present the modernization plan in ten minutes with costs and acceptance criteria; clarify responsibilities and domain expectations for the actual role."
                    ]
                }
            },
            pitfalls: {
                fresher: ["Poor time management in assessments", "Weak project explanation", "Overly generic HR answers"],
                experienced: ["No delivery governance examples", "Weak client impact metrics", "Insufficient domain clarity"]
            }
        },
        {
            name: "Infosys",
            group: "consulting",
            tag: "Foundational Screening + Delivery Focus",
            lastUpdated: "Preparation example; not independently verified",
            aliases: ["infy"],
            process: ["Campus/off-campus application", "Aptitude + coding test", "Technical round", "HR round", "Offer and onboarding"],
            rounds: [
                { name: "Test", check: "Reasoning, quant, coding basics", clear: "Maintain accuracy; confirm whether negative scoring applies instead of assuming it." },
                { name: "Technical", check: "Programming fundamentals and project contribution", clear: "Explain exact role played in each project." },
                { name: "HR", check: "Stability, communication, attitude", clear: "Be clear on career goals and willingness to learn." }
            ],
            application: {
                fresher: ["Focus on consistent aptitude practice and coding basics.", "Prepare project and internship explanation in simple language.", "Be ready for relocation and training expectations."],
                experienced: ["Highlight delivery consistency and client support outcomes.", "Show upskilling path in cloud/data/AI if switching projects.", "Prepare transition rationale clearly."]
            },
            guides: {
                fresher: {
                    focus: ["Aptitude + coding consistency", "Project ownership clarity", "Professional HR communication"],
                    day30: [
                        "Days 1-10: choose a language in [Programming languages]; implement a student-record model with validation and tests for missing/duplicate IDs.",
                        "Days 11-20: practice four short reasoning sets if relevant; revise DB keys, joins, and transactions by writing a student/course/enrollment example with sample rows.",
                        "Days 21-30: solve four basic scans in [DSA + design]; explain the student project's data flow and your own changes in two mock walkthroughs."
                    ],
                    day60: [
                        "Days 31-40: add enrollment rollback tests and explain why partial updates are unsafe; document one bug and its fix.",
                        "Days 41-50: run two timed practice assessments only using the confirmed format; record accuracy, timing, and follow-up fixes without assuming negative marking.",
                        "Days 51-60: rehearse project and STAR answers about learning and teamwork; ask for actual location/training/role terms rather than presuming flexibility is mandatory."
                    ]
                },
                experienced: {
                    focus: ["Delivery impact and reliability", "Client communication", "Upskilling evidence"],
                    day30: [
                        "Days 1-10: map the posting to your current stack; rewrite three resume bullets with verifiable delivery outcomes, not estimated achievements.",
                        "Days 11-20: trace one service request through network, application, and DB using [OS + networks]; document a real support incident's timeline.",
                        "Days 21-30: use [DSA + design] rollout lessons to plan an additive schema change; include compatibility tests and a rollback trigger."
                    ],
                    day60: [
                        "Days 31-40: build a minimal reproduction of one role-stack defect and test the fix; remove proprietary/customer data from all examples.",
                        "Days 41-50: run two project/stack-depth mocks and a client-escalation rehearsal; explain impact, containment, and prevention separately.",
                        "Days 51-60: rehearse a coherent transition story and a ten-minute project review; close two feedback gaps and confirm the assignment's actual scope."
                    ]
                }
            },
            pitfalls: {
                fresher: ["Inconsistent aptitude practice", "Unclear project contributions", "Low interview confidence"],
                experienced: ["No measurable delivery impact", "Weak upskilling narrative", "Confusing job-change reason"]
            }
        },
        {
            name: "TCS",
            group: "consulting",
            tag: "Volume Hiring + Fundamentals",
            lastUpdated: "Preparation example; not independently verified",
            aliases: ["tata consultancy"],
            process: ["Application", "NQT or role assessment", "Technical interview", "HR discussion", "Offer"],
            rounds: [
                { name: "NQT/Assessment", check: "Aptitude and coding readiness", clear: "Practice NQT pattern and speed-based solving." },
                { name: "Technical", check: "Core CS + language fundamentals", clear: "Answer basics confidently and show coding logic." },
                { name: "HR", check: "Communication and fit", clear: "Be clear, honest, and role-aligned." }
            ],
            application: {
                fresher: ["Prioritize NQT practice and fundamentals over advanced niche topics.", "Maintain clear, concise project documentation for interviews.", "Be prepared for role flexibility and learning commitments."],
                experienced: ["Highlight client delivery history, support handling, and SLA experience.", "Prepare transition story with domain continuity.", "Show certifications or recent upskilling evidence."]
            },
            guides: {
                fresher: {
                    focus: ["NQT performance", "Core programming fundamentals", "Communication clarity"],
                    day30: [
                        "Days 1-10: confirm whether this opportunity uses NQT (National Qualifier Test) or a different assessment; create a section/timing checklist from the invitation.",
                        "Days 11-20: use [Programming languages] for loops/functions and [DSA + design] for arrays; implement frequency counts and pair indices with empty/duplicate tests.",
                        "Days 21-30: complete three short section-wise drills in the confirmed format; prepare a project diagram and one example of a bug you personally fixed."
                    ],
                    day60: [
                        "Days 31-40: do two full self-timed assessment rehearsals if applicable; redo missed items after three days and record why the first method failed.",
                        "Days 41-50: practice a programming/DB fundamentals mock and a five-minute project explanation; answer what happens on invalid input.",
                        "Days 51-60: rehearse STAR teamwork and learning stories; compare skills against the specific role rather than assuming one test or interview path covers every TCS position."
                    ]
                },
                experienced: {
                    focus: ["Delivery discipline and reliability", "Client communication", "Modern stack adaptation"],
                    day30: [
                        "Days 1-10: select a target delivery/support/engineering role and map three actual accomplishments to its responsibilities.",
                        "Days 11-20: use [OS + networks] to trace a timeout incident; prepare evidence on detection, customer effect, root cause, and prevention.",
                        "Days 21-30: write a service-change plan in [DSA + design] with an SLO, compatibility checks, and a bounded rollback window."
                    ],
                    day60: [
                        "Days 31-40: implement or diagram one automation that reduces a repetitive support step; measure only what you can reproduce.",
                        "Days 41-50: run two project-depth mocks probing SLA (service-level agreement), domain constraints, and escalation decisions.",
                        "Days 51-60: revise two weak answers and rehearse a client update plus transition rationale; verify current stack, level, and interview stages with the recruiter."
                    ]
                }
            },
            pitfalls: {
                fresher: ["Ignoring NQT pattern", "Weak coding basics", "Rambling in interviews"],
                experienced: ["No SLA/customer impact metrics", "Weak modernization story", "Overly generic technical answers"]
            }
        },
        {
            name: "HCL",
            group: "consulting",
            tag: "Service Delivery + Technical Basics",
            lastUpdated: "Preparation example; not independently verified",
            aliases: ["hcltech"],
            process: ["Application", "Assessment", "Technical interview", "Managerial/HR", "Offer"],
            rounds: [
                { name: "Assessment", check: "Aptitude and coding logic", clear: "Strengthen logical reasoning and speed." },
                { name: "Technical", check: "Language, troubleshooting, projects", clear: "Show implementation depth and debugging mindset." },
                { name: "Managerial", check: "Team fit and delivery attitude", clear: "Emphasize ownership and accountability." }
            ],
            application: {
                fresher: ["Demonstrate coding fundamentals and eagerness to learn.", "Prepare concise answers around project modules and outcomes.", "Practice communication for support and delivery contexts."],
                experienced: ["Bring evidence of handling incidents, escalations, and delivery deadlines.", "Map skills to role stack requirements clearly.", "Prepare examples of process improvements and automation."]
            },
            guides: {
                fresher: {
                    focus: ["Assessment score", "Coding basics", "Project clarity"],
                    day30: [
                        "Days 1-10: use [Programming languages] to write a log-summary tool counting errors by code; test empty logs, malformed lines, and repeated codes.",
                        "Days 11-20: use [OS + networks] to explain process versus thread, DNS lookup, and HTTP status codes; reproduce one failed request locally.",
                        "Days 21-30: take two short timed reasoning/coding sets if invited; document the log tool and record a three-minute troubleshooting explanation."
                    ],
                    day60: [
                        "Days 31-40: add a configurable threshold and tests to the tool; write a runbook describing symptoms, checks, and safe recovery.",
                        "Days 41-50: rehearse two project/debugging mocks and a STAR example of asking for help early rather than hiding a blocker.",
                        "Days 51-60: simulate an incident handoff with evidence and next action; revise two weak fundamentals and check whether the actual role is support, operations, or development."
                    ]
                },
                experienced: {
                    focus: ["Operational ownership", "Troubleshooting examples", "Delivery leadership"],
                    day30: [
                        "Days 1-10: prepare two real incident timelines with detection, mitigation, root cause, and follow-up; redact sensitive details.",
                        "Days 11-20: use [OS + networks] to refresh connection failures, resource saturation, and timeout diagnosis; build one minimal failure reproduction.",
                        "Days 21-30: draft a runbook and an automation proposal with baseline manual effort and a testable improvement goal."
                    ],
                    day60: [
                        "Days 31-40: use reliability modules in [DSA + design] to size a bounded queue and define retry/backpressure behavior for a downstream outage.",
                        "Days 41-50: run two incident/stakeholder mocks; defend escalation timing, customer communication, and the trade-off behind your mitigation.",
                        "Days 51-60: present one operations improvement with real evidence and a rollback plan; confirm the target role's on-call/delivery expectations rather than assuming them."
                    ]
                }
            },
            pitfalls: {
                fresher: ["No coding confidence", "Weak project ownership", "Communication gaps"],
                experienced: ["No incident resolution examples", "Weak stakeholder communication stories", "No measurable improvements"]
            }
        },
        {
            name: "Capgemini",
            group: "consulting",
            tag: "Assessment + Domain Flexibility",
            lastUpdated: "Preparation example; not independently verified",
            aliases: ["capgemini"],
            process: ["Application", "Assessment", "Technical rounds", "HR round", "Offer"],
            rounds: [
                { name: "Assessment", check: "Aptitude, communication, coding", clear: "Simulate exam format end-to-end." },
                { name: "Technical", check: "Fundamentals + role-specific stack", clear: "Balance core theory and practical examples." },
                { name: "HR", check: "Culture fit and adaptability", clear: "Answer with flexibility and growth mindset." }
            ],
            application: {
                fresher: ["Build confidence in aptitude + coding + spoken communication.", "Keep resume role-focused and concise.", "Use project outcomes to show delivery mindset."],
                experienced: ["Show domain alignment and process maturity.", "Highlight project transitions and stakeholder coordination.", "Present quality and delivery metrics clearly."]
            },
            guides: {
                fresher: {
                    focus: ["Assessment discipline", "Core tech fundamentals", "Structured communication"],
                    day30: [
                        "Days 1-10: select the posting's language in [Programming languages] and build a ticket-status validator with allowed/rejected transitions.",
                        "Days 11-20: do three 25-minute reasoning/coding sets if relevant; test invalid transitions and explain the difference between validation and persistence.",
                        "Days 21-30: record a project explanation aimed at a client; include the business workflow, your module, and two acceptance tests."
                    ],
                    day60: [
                        "Days 31-40: extend the ticket model with a new optional field without breaking old records; use schema-evolution lessons in [DSA + design].",
                        "Days 41-50: rehearse two role-stack mocks and a stakeholder change-request scenario; list questions before promising delivery.",
                        "Days 51-60: run a mixed project/behavioral session and repair two logged weaknesses; confirm the assessment and domain needs for the particular vacancy."
                    ]
                },
                experienced: {
                    focus: ["Client-delivery evidence", "Technical depth for target role", "Cross-functional collaboration"],
                    day30: [
                        "Days 1-10: map three technical requirements to delivered work; prepare a project-transition story with the domain constraint and your learning plan.",
                        "Days 11-20: write an integration ADR using [DSA + design]; compare synchronous APIs with events, including failure recovery and ownership.",
                        "Days 21-30: review one target-stack defect with a minimal reproduction and regression tests; prepare a client-readable summary."
                    ],
                    day60: [
                        "Days 31-40: simulate an interface change affecting two teams; write compatibility tests, a migration sequence, and escalation owners.",
                        "Days 41-50: run two technical/domain mocks and one managerial scenario on conflicting priorities; use real outcomes rather than generic leadership phrases.",
                        "Days 51-60: revise the ADR using mock feedback and present a ten-minute delivery review; confirm role/domain fit and remaining gaps."
                    ]
                }
            },
            pitfalls: {
                fresher: ["Skipping communication prep", "Weak practice consistency", "Unclear role fit"],
                experienced: ["No quantifiable outcomes", "Weak domain story", "Limited adaptability examples"]
            }
        },
        {
            name: "Wipro",
            group: "consulting",
            tag: "Entry-Level Process + Delivery Orientation",
            lastUpdated: "Preparation example; not independently verified",
            aliases: ["wipro"],
            process: ["Application", "Assessment test", "Technical interview", "HR interview", "Offer"],
            rounds: [
                { name: "Test", check: "Aptitude and coding basics", clear: "Maintain accuracy and finish all easy sections." },
                { name: "Technical", check: "Programming, DBMS, project understanding", clear: "Focus on clarity over jargon." },
                { name: "HR", check: "Communication and professionalism", clear: "Show commitment and role understanding." }
            ],
            application: {
                fresher: ["Prepare test pattern and solve under strict time limits.", "Revise core concepts and 1-2 project narratives thoroughly.", "Practice simple and confident communication."],
                experienced: ["Highlight delivery consistency and project outcomes.", "Prepare stories around quality improvements and automation.", "Demonstrate team collaboration and ownership."]
            },
            guides: {
                fresher: {
                    focus: ["Assessment readiness", "CS fundamentals", "Project confidence"],
                    day30: [
                        "Days 1-10: use [Programming languages] for functions, collections, and exceptions; implement a record-import validator with malformed and empty-input tests.",
                        "Days 11-20: model import batches and rows with keys and a transaction boundary; explain why a failed batch should not leave half-applied data.",
                        "Days 21-30: complete two timed basic coding/reasoning sets if relevant and a five-minute project walkthrough; record errors and personal contributions."
                    ],
                    day60: [
                        "Days 31-40: use [DSA + design] to add a retry-safe batch ID; test resubmitting the same import and a conflicting changed payload.",
                        "Days 41-50: rehearse two technical/project mocks and a STAR story on improving quality after feedback.",
                        "Days 51-60: simulate the confirmed assessment format and redo failures after three days; prepare honest role/learning goals instead of a memorized HR script."
                    ]
                },
                experienced: {
                    focus: ["Delivery impact", "Support and maintenance maturity", "Upskilling continuity"],
                    day30: [
                        "Days 1-10: inventory three maintenance/delivery improvements with verifiable before/after evidence or explicitly qualitative results.",
                        "Days 11-20: reproduce one recurring defect in the target stack and write a regression test; explain containment versus permanent fix.",
                        "Days 21-30: use [DSA + design] for a safe API/schema rollout; document owner, monitoring, and rollback triggers."
                    ],
                    day60: [
                        "Days 31-40: build or specify one support automation with input validation and failure reporting; measure a reproducible task, not an invented productivity gain.",
                        "Days 41-50: run two technical/delivery mocks and an escalation rehearsal; practice customer impact and risk communication.",
                        "Days 51-60: rework two weak explanations and rehearse a project deep dive; check actual development/support responsibilities and selection stages."
                    ]
                }
            },
            pitfalls: {
                fresher: ["Under-preparing aptitude", "Weak fundamentals", "Unstructured answers"],
                experienced: ["No impact metrics", "Weak modernization story", "Unclear transition rationale"]
            }
        },
        {
            name: "Cognizant",
            group: "consulting",
            tag: "Role-Based Screening + Delivery Fit",
            lastUpdated: "Preparation example; not independently verified",
            aliases: ["cts", "cognizant"],
            process: ["Application", "Online assessment", "Technical interview", "HR/managerial", "Offer"],
            rounds: [
                { name: "Assessment", check: "Reasoning, coding, verbal", clear: "Prepare by section and maintain time discipline." },
                { name: "Technical", check: "Role stack and implementation ability", clear: "Use real project examples and troubleshooting approach." },
                { name: "HR", check: "Communication and role commitment", clear: "Stay concise and demonstrate professionalism." }
            ],
            application: {
                fresher: ["Target role-specific skills and keep project evidence practical.", "Practice coding basics plus communication rounds.", "Prepare location and shift flexibility answers if applicable."],
                experienced: ["Show measurable delivery outcomes and process ownership.", "Prepare examples of client communication and issue handling.", "Map resume to domain and role stack clearly."]
            },
            guides: {
                fresher: {
                    focus: ["Section-wise test prep", "Core fundamentals", "Interview communication"],
                    day30: [
                        "Days 1-10: choose one language in [Programming languages] and implement a small service-request tracker with valid status transitions and tests.",
                        "Days 11-20: practice three section-wise timed sets only for topics in the invitation; log reasoning, coding, and communication mistakes separately.",
                        "Days 21-30: prepare a request-flow diagram and explain one implementation bug; use [OS + networks] to trace a failed client/server call."
                    ],
                    day60: [
                        "Days 31-40: add validation and duplicate-request tests; use [DSA + design] maps to count request categories and explain complexity.",
                        "Days 41-50: run two project/stack mocks and rehearse a STAR example of coordinating with a teammate who had a different approach.",
                        "Days 51-60: simulate only the confirmed role's interview components, repair two failures, and prepare questions about assignment, stack, and any applicable shift requirements."
                    ]
                },
                experienced: {
                    focus: ["Delivery metrics", "Role-depth evidence", "Stakeholder stories"],
                    day30: [
                        "Days 1-10: map one target domain and stack to three real delivery accomplishments; explain how you know each result.",
                        "Days 11-20: trace a production issue through API, service, and DB; produce a minimal reproduction or sanitized sequence diagram.",
                        "Days 21-30: use [DSA + design] to specify one request workflow's SLO, retry budget, and rollout/rollback plan."
                    ],
                    day60: [
                        "Days 31-40: rehearse a client escalation caused by a missed dependency; write an update with facts, containment, owner, and next checkpoint.",
                        "Days 41-50: run two target-stack/deep-project mocks and one stakeholder scenario; document feedback and two concrete repairs.",
                        "Days 51-60: present a ten-minute ownership story linking technical action to actual client outcome; verify role-specific expectations instead of assuming every assignment shares a loop."
                    ]
                }
            },
            pitfalls: {
                fresher: ["Generic resumes", "Weak project confidence", "Inconsistent practice"],
                experienced: ["No client-impact evidence", "Shallow role depth", "Weak communication under probing"]
            }
        },
        {
            name: "Infor",
            group: "enterprise",
            tag: "Enterprise Product + Domain Context",
            lastUpdated: "Preparation example; not independently verified",
            aliases: ["infor"],
            process: ["Application and recruiter screening", "Technical rounds", "Domain/product fit discussion", "Managerial/HR", "Offer"],
            rounds: [
                { name: "Technical", check: "Core coding + system thinking", clear: "Balance correctness, maintainability, and practical constraints." },
                { name: "Product/Domain", check: "Enterprise use-case understanding", clear: "Relate solution to business workflow and reliability." },
                { name: "Behavioral", check: "Collaboration and accountability", clear: "Use examples with delivery ownership and impact." }
            ],
            application: {
                fresher: ["Show strong fundamentals and ability to learn domain workflows quickly.", "Include project cases relevant to enterprise workflows.", "Prepare SQL, APIs, and backend basics well."],
                experienced: ["Highlight enterprise-scale delivery and business-process understanding.", "Prepare stories on maintainability, SLA, and reliability.", "Show cross-functional collaboration with product/business teams."]
            },
            guides: {
                fresher: {
                    focus: ["Backend and data fundamentals", "Practical coding", "Domain adaptability"],
                    day30: [
                        "Days 1-10: model an inventory workflow with item, warehouse, and stock movement records; define keys, units, and two business invariants.",
                        "Days 11-20: implement a stock-adjustment API in your chosen [Programming languages] track; test missing items, duplicate requests, and insufficient stock.",
                        "Days 21-30: write sample SQL joins/aggregates over ten inventory rows; rehearse a five-minute explanation of how a user would detect a stock error."
                    ],
                    day60: [
                        "Days 31-40: use [DSA + design] API/storage lessons to make stock changes transactional and retry-safe; explain the concurrent-decrement case.",
                        "Days 41-50: run two backend/domain mocks and prepare a STAR story about learning a business rule from feedback.",
                        "Days 51-60: revise the inventory README with input/output examples and limitations; confirm the actual product/domain stack rather than treating the exercise as a hiring requirement."
                    ]
                },
                experienced: {
                    focus: ["Enterprise architecture choices", "Business process mapping", "Delivery reliability"],
                    day30: [
                        "Days 1-10: diagram an enterprise workflow you actually delivered; name its business invariants, stakeholders, and your responsibility.",
                        "Days 11-20: use [DSA + design] to compare transactional stock updates with asynchronous projections, including stale-read behavior and reconciliation.",
                        "Days 21-30: refresh four coding/SQL exercises tied to the target stack and prepare two maintainability/incident stories with real evidence."
                    ],
                    day60: [
                        "Days 31-40: propose a nonbreaking workflow extension with migration and rollback; state which invariants must hold during partial failure.",
                        "Days 41-50: run two architecture/domain mocks and a product-priority conflict rehearsal; defend cost and reliability trade-offs.",
                        "Days 51-60: present a ten-minute enterprise project review and repair two mock weaknesses; ask which product area, domain, and role depth apply."
                    ]
                }
            },
            pitfalls: {
                fresher: ["Ignoring domain context", "Weak SQL/API knowledge", "No production mindset"],
                experienced: ["No business process mapping", "Lack of maintainability examples", "Weak stakeholder alignment stories"]
            }
        },
        {
            name: "Salesforce",
            group: "enterprise",
            tag: "Cloud Platform + CRM Ecosystem",
            lastUpdated: "Preparation example; not independently verified",
            aliases: ["agentforce", "sf"],
            process: ["Application/referral", "Recruiter call", "Technical rounds", "System/product design depending level", "Behavioral and offer"],
            rounds: [
                { name: "Coding/Platform", check: "Programming logic and platform thinking", clear: "Demonstrate clean design and ecosystem awareness." },
                { name: "Design", check: "Scalable product architecture", clear: "Discuss tenancy, integration, and reliability trade-offs." },
                { name: "Behavioral", check: "Customer-first mindset", clear: "Use customer impact examples with measurable outcomes." }
            ],
            application: {
                fresher: ["Build project stories around APIs, integrations, and cloud workflows.", "Show quick learning capability for platform ecosystems.", "Strengthen OOP and DB fundamentals."],
                experienced: ["Demonstrate product and platform architecture ownership.", "Highlight multi-system integration and data consistency decisions.", "Prepare leadership examples around product quality and customer outcomes."]
            },
            guides: {
                fresher: {
                    focus: ["Core coding fundamentals", "Cloud and API basics", "Customer-centric communication"],
                    day30: [
                        "Days 1-10: distinguish a platform-specific role from general software engineering in the posting; choose the relevant language and OOP modules in [Programming languages].",
                        "Days 11-20: build a mock contact-sync adapter with a local dataset; test missing identifiers, duplicate delivery, and failed updates without using customer data.",
                        "Days 21-30: practice four array/map exercises in [DSA + design]; record a project walkthrough explaining user value and one integration limitation."
                    ],
                    day60: [
                        "Days 31-40: add idempotency and per-tenant data isolation to the mock adapter; explain why one tenant must never read another's records.",
                        "Days 41-50: rehearse two coding/integration mocks and two STAR stories on collaboration and responding to user feedback.",
                        "Days 51-60: run a mixed mock and repair two failures; study platform-specific APIs only if named by the actual role, not merely because the company sells a platform."
                    ]
                },
                experienced: {
                    focus: ["Platform architecture depth", "Data/integration trade-offs", "Cross-team leadership"],
                    day30: [
                        "Days 1-10: map the posting to platform development, product engineering, or another specialty; list three matching evidence-backed achievements.",
                        "Days 11-20: design a multi-tenant contact-sync service using [DSA + design]; define tenant keys, quotas, retry semantics, and conflict policy.",
                        "Days 21-30: prepare four real stories on integration failure, customer impact, cross-team influence, and a rejected design alternative."
                    ],
                    day60: [
                        "Days 31-40: stress the design with one hot tenant and downstream throttling; specify isolation, backpressure, and measured success criteria.",
                        "Days 41-50: run two architecture/integration mocks and two coding refresh screens; add a regression test for each missed failure case.",
                        "Days 51-60: rehearse a customer-facing project review and update the design ADR; confirm platform-specific knowledge and level expectations for the actual position."
                    ]
                }
            },
            pitfalls: {
                fresher: ["Weak API/integration understanding", "No customer-impact narrative", "Unclear project depth"],
                experienced: ["Generic design answers", "Weak platform constraints discussion", "No leadership evidence"]
            }
        },
        {
            name: "SAP",
            group: "enterprise",
            tag: "Enterprise Platforms + Business Process Engineering",
            lastUpdated: "Preparation example; not independently verified",
            aliases: ["sap"],
            process: ["Application screening", "Technical assessments/interviews", "Domain + architecture discussion", "Behavioral/managerial", "Offer"],
            rounds: [
                { name: "Technical", check: "Programming fundamentals and integration concepts", clear: "Keep answers practical and enterprise-oriented." },
                { name: "Domain", check: "Business workflow understanding", clear: "Tie technical decisions to business process value." },
                { name: "Behavioral", check: "Collaboration across functions", clear: "Show stakeholder alignment and delivery ownership." }
            ],
            application: {
                fresher: ["Highlight willingness to learn domain-heavy enterprise stacks.", "Show SQL, backend, and integration basics through projects.", "Prepare clear explanation for business problem solved."],
                experienced: ["Demonstrate enterprise transformation or integration outcomes.", "Discuss architecture decisions for compliance, reliability, and scale.", "Show partnership with functional/business teams."]
            },
            guides: {
                fresher: {
                    focus: ["Backend + DB confidence", "Domain learning agility", "Structured communication"],
                    day30: [
                        "Days 1-10: model a purchase-order workflow from created to approved to received; distinguish application development from functional/domain roles in the posting.",
                        "Days 11-20: use [Programming languages] to implement allowed transitions with rejected-state tests; write sample SQL for pending orders and line totals.",
                        "Days 21-30: explain one workflow to a nontechnical listener and prepare a project diagram showing your contribution and one business invariant."
                    ],
                    day60: [
                        "Days 31-40: use [DSA + design] transaction/API lessons to handle duplicate approval requests and concurrent edits; document expected conflict behavior.",
                        "Days 41-50: run two programming/domain mocks and rehearse a STAR story about reconciling differing stakeholder requirements.",
                        "Days 51-60: revise the purchase-order demo with invalid-input and rollback tests; study a specific enterprise language/platform only if the vacancy calls for it."
                    ]
                },
                experienced: {
                    focus: ["Enterprise design depth", "Business-tech translation", "Program delivery ownership"],
                    day30: [
                        "Days 1-10: map the role to a business process and technical/functional scope; catalogue three real transformation or integration decisions.",
                        "Days 11-20: use [DSA + design] to draft an order-workflow migration with data reconciliation, compatibility checks, and rollback boundaries.",
                        "Days 21-30: prepare stories involving business/engineering disagreement, reliability, and delivery ownership; separate observed outcomes from estimates."
                    ],
                    day60: [
                        "Days 31-40: simulate duplicate business events and a partial migration failure; state which records are authoritative and how to reconcile.",
                        "Days 41-50: run two architecture/domain mocks and a stakeholder review; explain one compliance constraint without claiming unsupported expertise.",
                        "Days 51-60: present a ten-minute transformation review with real impact evidence and unresolved risks; verify product stack and responsibilities before targeted final revision."
                    ]
                }
            },
            pitfalls: {
                fresher: ["Pure tech answers without business angle", "Weak SQL/integration basics", "No structured examples"],
                experienced: ["No transformation metrics", "Weak compliance/reliability discussion", "Insufficient functional collaboration evidence"]
            }
        },
        {
            name: "IBM",
            group: "enterprise",
            tag: "Hybrid Cloud + AI + Enterprise Delivery",
            lastUpdated: "Preparation example; not independently verified",
            aliases: ["watson"],
            process: ["Application + screening", "Assessment or coding round", "Technical and system discussions", "Behavioral/managerial", "Offer"],
            rounds: [
                { name: "Assessment", check: "Problem solving and fundamentals", clear: "Show methodical approach and clean logic." },
                { name: "Technical", check: "Coding, cloud/AI basics, project depth", clear: "Explain design choices and implementation trade-offs." },
                { name: "Managerial", check: "Collaboration and execution", clear: "Highlight accountability and measurable delivery outcomes." }
            ],
            application: {
                fresher: ["Build a practical project around cloud, automation, or AI workflow.", "Strengthen coding fundamentals and communication.", "Show curiosity and learning consistency."],
                experienced: ["Highlight modernization and platform migration outcomes.", "Discuss architecture and operations trade-offs.", "Prepare leadership stories on delivery and quality improvements."]
            },
            guides: {
                fresher: {
                    focus: ["Coding and CS fundamentals", "Practical cloud/AI understanding", "Interview communication"],
                    day30: [
                        "Days 1-10: choose a role-aligned language in [Programming languages]; build a local job-status tool with queued/running/succeeded/failed states and tests.",
                        "Days 11-20: use [OS + networks] to trace process failure, networking timeout, and a retry; distinguish infrastructure issues from application bugs.",
                        "Days 21-30: solve four scans/traversal exercises in [DSA + design]; document your job tool and rehearse a five-minute project explanation."
                    ],
                    day60: [
                        "Days 31-40: design an asynchronous worker and retry budget for the tool using [DSA + design]; test duplicate events and permanent failures.",
                        "Days 41-50: rehearse two coding/project mocks and a STAR learning story; discuss AI topics only if relevant to the actual posting and your evidence.",
                        "Days 51-60: run a mixed mock, redo two failed exercises after three days, and confirm whether the role centers on software, infrastructure, consulting, or another specialty."
                    ]
                },
                experienced: {
                    focus: ["Architecture and modernization depth", "Cross-team execution", "Business impact articulation"],
                    day30: [
                        "Days 1-10: map a modernization/platform posting to three actual accomplishments; distinguish your contribution from the broader program.",
                        "Days 11-20: use [DSA + design] to draft a staged service migration with numeric load, dependency boundaries, reconciliation, and rollback.",
                        "Days 21-30: review a real incident and a cross-team design disagreement; refresh four coding/debugging problems in the target stack."
                    ],
                    day60: [
                        "Days 31-40: model a network partition and downstream outage with [OS + networks] and reliability lessons; document data-loss/recovery assumptions.",
                        "Days 41-50: run two architecture/operations mocks and a stakeholder review; require explicit costs and failure semantics for each major decision.",
                        "Days 51-60: present a ten-minute modernization review with real evidence, repair two recurring gaps, and confirm role-specific technology and interview expectations."
                    ]
                }
            },
            pitfalls: {
                fresher: ["Weak coding consistency", "Shallow project depth", "No clear learning narrative"],
                experienced: ["No modernization impact metrics", "Weak architecture communication", "Generic leadership examples"]
            }
        }
    ];

    var activeAudience = "fresher";
    var activeGroup = "all";
    var activeQuery = "";
    var groupLabels = {
        product: "Product MNC",
        consulting: "Consulting & Services",
        enterprise: "Enterprise Tech"
    };

    var companyLogoFile = {
        "Microsoft": "microsoft.svg",
        "Amazon": "amazon.svg",
        "Google": "google.svg",
        "Deloitte": "deloitte.svg",
        "Accenture": "accenture.svg",
        "Infosys": "infosys.svg",
        "TCS": "tcs.svg",
        "HCL": "hcltech.svg",
        "Capgemini": "capgemini.svg",
        "Wipro": "wipro.svg",
        "Cognizant": "cognizant.svg",
        "Infor": "infor.svg",
        "Salesforce": "salesforce.svg",
        "SAP": "sap.svg",
        "IBM": "ibm.svg"
    };

    var companyAudienceContent = {
        "Microsoft": {
            process: {
                fresher: [
                    "Application/referral screening",
                    "Online assessment or campus shortlisting",
                    "2-3 coding + CS fundamentals rounds",
                    "Behavioral or hiring-manager round",
                    "Offer and team matching"
                ],
                experienced: [
                    "Recruiter calibration and resume screening",
                    "Coding screen or direct shortlist by profile",
                    "Coding + system design + project deep dive loop",
                    "Managerial/collaboration interview",
                    "Hiring decision, team match, and offer"
                ]
            },
            rounds: {
                fresher: [
                    { name: "Coding", check: "DSA basics, problem solving, clean code", clear: "Explain approach before coding and validate with examples." },
                    { name: "CS Fundamentals", check: "OS, DBMS, networking, OOP basics", clear: "Answer from first principles with simple real examples." },
                    { name: "Behavioral", check: "Learning ability, teamwork, initiative", clear: "Use concise STAR stories from projects, internships, or clubs." }
                ],
                experienced: [
                    { name: "Coding / Debugging", check: "Implementation quality, edge cases, code clarity", clear: "Balance speed with production-style reasoning and test coverage." },
                    { name: "System Design", check: "Scalability, APIs, trade-offs, reliability", clear: "Start from requirements and justify architecture choices with metrics." },
                    { name: "Project + Behavioral", check: "Ownership, impact, cross-team execution", clear: "Show measurable outcomes, failures, and decision trade-offs." }
                ]
            }
        },
        "Amazon": {
            process: {
                fresher: [
                    "Application and recruiter connect",
                    "Online assessment",
                    "1-2 technical interviews",
                    "Leadership-principle + final loop",
                    "Offer discussion"
                ],
                experienced: [
                    "Recruiter calibration and profile review",
                    "Coding or technical phone screen",
                    "Virtual onsite: coding + design + debugging",
                    "Bar raiser and leadership-principle deep dive",
                    "Hiring decision and compensation discussion"
                ]
            },
            rounds: {
                fresher: [
                    { name: "OA", check: "Timed coding accuracy and speed", clear: "Secure easy-to-medium questions first, then optimize." },
                    { name: "Technical", check: "DSA, CS basics, implementation clarity", clear: "Think aloud and keep communication structured." },
                    { name: "LP Behavioral", check: "Ownership, bias for action, learning", clear: "Map each answer to a clear leadership principle." }
                ],
                experienced: [
                    { name: "Coding / Debugging", check: "Hands-on implementation under probing", clear: "State assumptions, discuss complexity, and test failure cases." },
                    { name: "System Design", check: "Scale, resiliency, and operational trade-offs", clear: "Ground design in scale estimates and monitoring decisions." },
                    { name: "Bar Raiser + LP", check: "Judgment, ownership, difficult decisions", clear: "Use high-stakes stories with metrics and failure recovery." }
                ]
            }
        },
        "Google": {
            process: {
                fresher: [
                    "Application/referral screening",
                    "Recruiter conversation",
                    "2-3 coding interviews",
                    "Googlyness/behavioral round",
                    "Hiring committee and team match"
                ],
                experienced: [
                    "Recruiter calibration and role mapping",
                    "Technical screen",
                    "3-5 interviews across coding, design, and collaboration",
                    "Hiring committee review",
                    "Team match and offer"
                ]
            },
            rounds: {
                fresher: [
                    { name: "Coding", check: "Algorithmic thinking and clear communication", clear: "Move cleanly from brute force to optimized approach." },
                    { name: "Problem Solving", check: "Reasoning depth and fundamentals", clear: "Clarify constraints before coding and validate examples." },
                    { name: "Googlyness", check: "Humility, collaboration, learning mindset", clear: "Use examples that show initiative without ego." }
                ],
                experienced: [
                    { name: "Coding", check: "Correctness, code quality, edge-case handling", clear: "Keep explanations crisp and evaluate alternatives." },
                    { name: "System Design", check: "Architecture, trade-offs, scale reasoning", clear: "Use requirement-first thinking and discuss costs and bottlenecks." },
                    { name: "Leadership / Googlyness", check: "Influence, ambiguity handling, collaboration", clear: "Show impact across teams with concrete outcomes." }
                ]
            }
        },
        "Deloitte": {
            process: {
                fresher: [
                    "Application screening",
                    "Aptitude or technical assessment",
                    "Technical interview",
                    "HR or managerial round",
                    "Offer and onboarding"
                ],
                experienced: [
                    "Profile screening for role and client fit",
                    "Technical/domain interview",
                    "Project delivery and client-readiness discussion",
                    "Managerial/stakeholder round",
                    "Offer and deployment alignment"
                ]
            },
            rounds: {
                fresher: [
                    { name: "Assessment", check: "Aptitude, reasoning, basic role readiness", clear: "Time-box sections and protect accuracy." },
                    { name: "Technical", check: "Fundamentals, project basics, communication", clear: "Explain project implementation in simple business terms." },
                    { name: "HR / Communication", check: "Professionalism and client-facing potential", clear: "Answer clearly, with structure and confidence." }
                ],
                experienced: [
                    { name: "Domain + Delivery", check: "Project ownership, timelines, delivery constraints", clear: "Use examples with deadlines, scope, and client outcomes." },
                    { name: "Client / Stakeholder", check: "Requirement handling, escalation, communication", clear: "Show calm judgment and stakeholder awareness." },
                    { name: "Managerial", check: "Leadership, mentorship, accountability", clear: "Present measurable execution impact and team influence." }
                ]
            }
        },
        "Accenture": {
            process: {
                fresher: [
                    "Application",
                    "Cognitive and coding assessments",
                    "Technical interview",
                    "HR round",
                    "Offer"
                ],
                experienced: [
                    "Profile screening and role mapping",
                    "Assessment or screening discussion",
                    "Technical + scenario-based interview",
                    "Managerial/client-fit round",
                    "Offer"
                ]
            },
            rounds: {
                fresher: [
                    { name: "Assessments", check: "Aptitude, coding basics, communication", clear: "Practice full test format and maintain pacing." },
                    { name: "Technical", check: "Stack fundamentals and project explanation", clear: "Use clear module ownership and implementation flow." },
                    { name: "HR", check: "Adaptability and growth mindset", clear: "Use examples of quick learning and teamwork." }
                ],
                experienced: [
                    { name: "Technical Scenario", check: "Delivery choices, modernization, risk handling", clear: "Tie technical answers to project outcomes and client value." },
                    { name: "Managerial", check: "Stakeholder handling and delivery discipline", clear: "Show process maturity, communication, and escalation control." },
                    { name: "Role Fit", check: "Domain alignment and leadership readiness", clear: "Position experience directly against the target role." }
                ]
            }
        },
        "Infosys": {
            process: {
                fresher: [
                    "Campus/off-campus application",
                    "Aptitude and coding test",
                    "Technical round",
                    "HR round",
                    "Offer and onboarding"
                ],
                experienced: [
                    "Resume screening and recruiter connect",
                    "Technical screening",
                    "Project and delivery deep dive",
                    "Managerial/HR alignment",
                    "Offer"
                ]
            },
            rounds: {
                fresher: [
                    { name: "Test", check: "Reasoning, quant, coding basics", clear: "Protect sectional accuracy and avoid careless errors." },
                    { name: "Technical", check: "Programming fundamentals and project contribution", clear: "Be specific about your role in each project." },
                    { name: "HR", check: "Communication, stability, willingness to learn", clear: "Answer simply and stay role-aligned." }
                ],
                experienced: [
                    { name: "Technical", check: "Role stack depth and implementation choices", clear: "Discuss delivery trade-offs, support issues, and improvements." },
                    { name: "Delivery", check: "Client outcomes, reliability, project ownership", clear: "Use measurable examples of consistency and accountability." },
                    { name: "Managerial", check: "Transition rationale and future fit", clear: "Keep your career story coherent and credible." }
                ]
            }
        },
        "TCS": {
            process: {
                fresher: [
                    "Application",
                    "NQT or role assessment",
                    "Technical interview",
                    "HR discussion",
                    "Offer"
                ],
                experienced: [
                    "Profile screening",
                    "Technical screening or panel discussion",
                    "Project and client-delivery interview",
                    "Managerial/HR discussion",
                    "Offer"
                ]
            },
            rounds: {
                fresher: [
                    { name: "NQT / Assessment", check: "Aptitude and basic coding", clear: "Practice the exact pattern and manage time carefully." },
                    { name: "Technical", check: "Core CS and programming basics", clear: "Answer fundamentals confidently with simple examples." },
                    { name: "HR", check: "Communication, flexibility, fit", clear: "Stay concise and show willingness to learn." }
                ],
                experienced: [
                    { name: "Technical", check: "Role stack, support/delivery depth, troubleshooting", clear: "Use recent project examples with incident ownership." },
                    { name: "Client Delivery", check: "SLA awareness, customer handling, execution discipline", clear: "Show reliability and measurable service outcomes." },
                    { name: "Managerial", check: "Role transition and team fit", clear: "Position yourself with a clear continuity story." }
                ]
            }
        },
        "HCL": {
            process: {
                fresher: [
                    "Application",
                    "Assessment",
                    "Technical interview",
                    "Managerial/HR",
                    "Offer"
                ],
                experienced: [
                    "Profile screening",
                    "Technical screening",
                    "Project, troubleshooting, and operations discussion",
                    "Managerial round",
                    "Offer"
                ]
            },
            rounds: {
                fresher: [
                    { name: "Assessment", check: "Aptitude and coding logic", clear: "Prioritize accuracy and reasoning discipline." },
                    { name: "Technical", check: "Language basics, projects, debugging", clear: "Explain implementation details, not just theory." },
                    { name: "Managerial", check: "Delivery attitude and communication", clear: "Show ownership and willingness to learn." }
                ],
                experienced: [
                    { name: "Technical", check: "Troubleshooting depth and stack ownership", clear: "Discuss incidents, root cause analysis, and fixes." },
                    { name: "Operations / Delivery", check: "Escalations, deadlines, process improvement", clear: "Show measurable reliability and automation impact." },
                    { name: "Managerial", check: "Team fit and leadership readiness", clear: "Use accountability and stakeholder-management stories." }
                ]
            }
        },
        "Capgemini": {
            process: {
                fresher: [
                    "Application",
                    "Assessment",
                    "Technical rounds",
                    "HR round",
                    "Offer"
                ],
                experienced: [
                    "Profile screening",
                    "Technical screening or assessment",
                    "Role-depth and domain interview",
                    "Managerial round",
                    "Offer"
                ]
            },
            rounds: {
                fresher: [
                    { name: "Assessment", check: "Aptitude, coding, communication", clear: "Simulate the full format and practice consistency." },
                    { name: "Technical", check: "Core fundamentals and role basics", clear: "Balance theory with practical examples." },
                    { name: "HR", check: "Culture fit and flexibility", clear: "Show adaptability and a learning mindset." }
                ],
                experienced: [
                    { name: "Role Technical", check: "Target-stack depth and project transitions", clear: "Explain how your experience fits the required role." },
                    { name: "Delivery / Domain", check: "Client impact, quality, coordination", clear: "Use quantifiable outcomes and cross-team examples." },
                    { name: "Managerial", check: "Stakeholder fit and leadership maturity", clear: "Show calm communication and ownership." }
                ]
            }
        },
        "Wipro": {
            process: {
                fresher: [
                    "Application",
                    "Assessment test",
                    "Technical interview",
                    "HR interview",
                    "Offer"
                ],
                experienced: [
                    "Resume screening",
                    "Technical screening",
                    "Project and delivery discussion",
                    "Managerial/HR round",
                    "Offer"
                ]
            },
            rounds: {
                fresher: [
                    { name: "Test", check: "Aptitude and coding basics", clear: "Finish easy sections first and avoid panic." },
                    { name: "Technical", check: "Programming, DBMS, project understanding", clear: "Keep explanations simple and accurate." },
                    { name: "HR", check: "Professionalism and communication", clear: "Stay structured, polite, and role-aware." }
                ],
                experienced: [
                    { name: "Technical", check: "Delivery stack depth and practical troubleshooting", clear: "Show how you improved quality, speed, or reliability." },
                    { name: "Project / Delivery", check: "Ownership, maintenance maturity, team collaboration", clear: "Use real examples with metrics and outcomes." },
                    { name: "Managerial", check: "Career alignment and stakeholder fit", clear: "Keep your transition story clear and grounded." }
                ]
            }
        },
        "Cognizant": {
            process: {
                fresher: [
                    "Application",
                    "Online assessment",
                    "Technical interview",
                    "HR/managerial round",
                    "Offer"
                ],
                experienced: [
                    "Profile screening",
                    "Technical screening",
                    "Role-depth and client-delivery discussion",
                    "Managerial round",
                    "Offer"
                ]
            },
            rounds: {
                fresher: [
                    { name: "Assessment", check: "Reasoning, coding, verbal ability", clear: "Prepare section-wise and keep steady timing." },
                    { name: "Technical", check: "Fundamentals and project implementation", clear: "Use practical examples and avoid buzzwords." },
                    { name: "HR", check: "Communication and commitment", clear: "Be concise and confident about role fit." }
                ],
                experienced: [
                    { name: "Technical", check: "Role stack depth and troubleshooting ability", clear: "Show implementation ownership, not just participation." },
                    { name: "Client / Delivery", check: "Process ownership, client communication, outcomes", clear: "Use metrics and stakeholder examples." },
                    { name: "Managerial", check: "Leadership readiness and continuity", clear: "Present a coherent career path and impact story." }
                ]
            }
        },
        "Infor": {
            process: {
                fresher: [
                    "Application and recruiter screening",
                    "Technical coding round",
                    "Product/domain fit discussion",
                    "Behavioral or managerial round",
                    "Offer"
                ],
                experienced: [
                    "Recruiter calibration and role screening",
                    "Coding or architecture discussion",
                    "Domain and enterprise product deep dive",
                    "Managerial/cross-functional round",
                    "Offer"
                ]
            },
            rounds: {
                fresher: [
                    { name: "Technical", check: "Coding basics, SQL, API thinking", clear: "Keep answers practical and enterprise-oriented." },
                    { name: "Product / Domain", check: "Workflow understanding and business context", clear: "Tie solutions to real business use cases." },
                    { name: "Behavioral", check: "Ownership and learning ability", clear: "Show accountability and adaptability from projects." }
                ],
                experienced: [
                    { name: "Technical / Architecture", check: "Maintainability, design quality, scale constraints", clear: "Explain trade-offs with reliability and business impact." },
                    { name: "Domain Deep Dive", check: "Enterprise workflows and product reasoning", clear: "Translate technical choices into business process outcomes." },
                    { name: "Managerial", check: "Cross-functional execution and delivery ownership", clear: "Use examples involving product, business, and engineering alignment." }
                ]
            }
        },
        "Salesforce": {
            process: {
                fresher: [
                    "Application/referral",
                    "Recruiter call",
                    "Technical rounds",
                    "Behavioral discussion",
                    "Offer"
                ],
                experienced: [
                    "Recruiter calibration",
                    "Technical screen",
                    "Coding, design, and platform discussion loop",
                    "Leadership/behavioral round",
                    "Offer"
                ]
            },
            rounds: {
                fresher: [
                    { name: "Coding / Platform", check: "Programming logic and platform awareness", clear: "Demonstrate clean thinking and API fundamentals." },
                    { name: "Product Thinking", check: "Customer-centric reasoning and cloud basics", clear: "Connect design choices to user and business value." },
                    { name: "Behavioral", check: "Collaboration and learning mindset", clear: "Use examples with ownership and customer impact." }
                ],
                experienced: [
                    { name: "Coding / Platform", check: "Implementation depth and platform trade-offs", clear: "Discuss APIs, data flows, and reliability clearly." },
                    { name: "System Design", check: "Scale, integration, tenancy, and consistency", clear: "Anchor design in constraints and operational needs." },
                    { name: "Leadership", check: "Customer-first execution and influence", clear: "Show measurable product impact across teams." }
                ]
            }
        },
        "SAP": {
            process: {
                fresher: [
                    "Application screening",
                    "Technical assessments/interviews",
                    "Domain discussion",
                    "Behavioral/managerial round",
                    "Offer"
                ],
                experienced: [
                    "Profile screening",
                    "Technical screening",
                    "Architecture and business-process deep dive",
                    "Stakeholder/managerial round",
                    "Offer"
                ]
            },
            rounds: {
                fresher: [
                    { name: "Technical", check: "Programming, SQL, integration fundamentals", clear: "Keep answers practical and tie them to use cases." },
                    { name: "Domain", check: "Business workflow understanding", clear: "Relate technical choices to process value." },
                    { name: "Behavioral", check: "Collaboration and structured communication", clear: "Show clarity, discipline, and learning speed." }
                ],
                experienced: [
                    { name: "Technical / Architecture", check: "Enterprise design, integration, reliability", clear: "Discuss transformation trade-offs and long-term maintainability." },
                    { name: "Business Process", check: "Functional alignment and domain reasoning", clear: "Translate architecture decisions into business outcomes." },
                    { name: "Stakeholder", check: "Program ownership and cross-functional collaboration", clear: "Use examples spanning tech and business teams." }
                ]
            }
        },
        "IBM": {
            process: {
                fresher: [
                    "Application and screening",
                    "Assessment or coding round",
                    "Technical interview",
                    "Behavioral/managerial round",
                    "Offer"
                ],
                experienced: [
                    "Recruiter screening and role alignment",
                    "Technical screen",
                    "Coding, architecture, and modernization discussion",
                    "Managerial/leadership round",
                    "Offer"
                ]
            },
            rounds: {
                fresher: [
                    { name: "Assessment", check: "Problem solving and CS fundamentals", clear: "Use a methodical approach and narrate your reasoning." },
                    { name: "Technical", check: "Coding, cloud/AI basics, project depth", clear: "Explain design choices and implementation details clearly." },
                    { name: "Behavioral", check: "Learning agility and execution mindset", clear: "Show curiosity, discipline, and accountability." }
                ],
                experienced: [
                    { name: "Technical", check: "Architecture depth, modernization, operations trade-offs", clear: "Use recent projects with concrete metrics and constraints." },
                    { name: "Project / Design", check: "Hybrid cloud, AI, reliability decisions", clear: "Show system thinking, scale reasoning, and delivery realism." },
                    { name: "Managerial", check: "Cross-team execution and leadership", clear: "Present impact through influence, delivery, and measurable outcomes." }
                ]
            }
        }
    };

    function escapeHtml(text) {
        return String(text)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/\"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function buildList(items) {
        return "<ul class=\"hiring-list\">" + items.map(function (item) {
            return "<li>" + escapeHtml(item) + "</li>";
        }).join("") + "</ul>";
    }

    var studyLinks = {
        "DSA + design": "courses.html?track=track-dsa-sd",
        "OS + networks": "courses.html?track=track-os-net",
        "Programming languages": "courses.html?track=track-programming-languages"
    };

    function buildStudyList(items) {
        return "<ul class=\"hiring-list\">" + items.map(function (item) {
            var text = escapeHtml(item).replace(/\[([^\]]+)\]/g, function (token, label) {
                if (!Object.prototype.hasOwnProperty.call(studyLinks, label)) return token;
                return "<a href=\"" + escapeHtml(studyLinks[label]) + "\">" + label + "</a>";
            });
            return "<li>" + text + "</li>";
        }).join("") + "</ul>";
    }

    function buildPreparationContext(audience) {
        var roleScope = audience === "fresher"
            ? "Early-career candidates can use coursework, clubs, personal projects, internships, or volunteer work as evidence; production experience is not assumed."
            : "Experienced candidates should calibrate depth to the target scope: explain actual design decisions, delivery responsibility, incident handling, and influence with evidence.";
        return [
            "<p>These are illustrative preparation routes, not verified company hiring requirements. Stages, order, number of interviews, assessments, and timelines vary by role, level, team, location, and hiring program; no step or outcome is guaranteed. Confirm details in the official job posting, application portal, or recruiter invitation.</p>",
            "<p>Software engineering, support, data, consulting, functional, and platform-specific roles can emphasize very different skills. A company's products do not mean every candidate needs the same stack. ",
            escapeHtml(roleScope), "</p>",
            "<p>OA = online assessment. LP = leadership principles (used here as an Amazon-oriented behavioral preparation lens, not a separate round guaranteed for every role). STAR = Situation, Task, Action, Result. NQT = National Qualifier Test; confirm whether it applies to the particular TCS opportunity.</p>",
            "<p>The 30/60-day schedule is a suggested study allocation, not an expected hiring timeline: aim for five 45-60 minute sessions per week, reduce scope if needed, and use the final session to test and review. Course links open existing tracks; select the named module there. Role-specific SQL/domain exercises are practice tasks, not claims of dedicated course coverage.</p>"
        ].join("");
    }

    function buildBehavioralPractice(audience) {
        var example = audience === "fresher"
            ? "Illustrative fresher answer, not a script to claim as your experience: Situation: our three-person college team repeatedly lost demo submissions. Task: I owned making the form reliable before the next review. Action: I reproduced an empty-field failure, added validation and three regression cases, and asked a teammate to review the change; we agreed on a smaller feature scope. Result: all three cases passed in our next demo. I learned to validate assumptions early; next time I would add the checks before integration."
            : "Use a real project: Situation: describe the incident/change and constraints without confidential details. Task: state your responsibility and the success criterion. Action: explain your diagnosis, alternatives, personal decision, collaboration, and rollout. Result: report an observed outcome and how it was measured; include a limitation and what you would change. If no numerical metric exists, give a verifiable qualitative result rather than inventing one.";
        return [
            "<p>Prompt: Tell me about a setback or disagreement and what you changed. Spend about 15 seconds on Situation, 10 on Task, 45 on Action, and 20 on Result; adapt to follow-up questions.</p>",
            "<p>", escapeHtml(example), "</p>",
            "<p>Practice: write two truthful stories, one about teamwork and one about a mistake. Label S/T/A/R, separate your actions from team actions, and rehearse: Why this choice? What evidence supports the result? What would you do differently? Never borrow the illustrative outcome as your own.</p>"
        ].join("");
    }

    function buildDisclosure(title, icon, bodyHtml, isOpen) {
        return [
            "<details class=\"hiring-disclosure\"" + (isOpen ? " open" : "") + ">",
            "<summary><i class=\"fas " + escapeHtml(icon) + "\" aria-hidden=\"true\"></i><span>" + escapeHtml(title) + "</span></summary>",
            "<div class=\"hiring-disclosure-body\">" + bodyHtml + "</div>",
            "</details>"
        ].join("");
    }

    function getCompanyCode(name) {
        var parts = String(name).trim().split(/\s+/);
        if (parts.length === 1) {
            return parts[0].slice(0, 3).toUpperCase();
        }
        return parts.slice(0, 2).map(function (p) { return p.charAt(0).toUpperCase(); }).join("");
    }

    function getCompanyLogoUrl(name) {
        var file = companyLogoFile[name];
        if (!file) return "";
        return "assets/logos/" + file;
    }

    function getAudienceProcess(company) {
        var audienceContent = companyAudienceContent[company.name];
        return audienceContent && audienceContent.process && audienceContent.process[activeAudience]
            ? audienceContent.process[activeAudience]
            : company.process;
    }

    function getAudienceRounds(company) {
        var audienceContent = companyAudienceContent[company.name];
        return audienceContent && audienceContent.rounds && audienceContent.rounds[activeAudience]
            ? audienceContent.rounds[activeAudience]
            : company.rounds;
    }

    function updateResultsBar(total, shown) {
        var bar = document.getElementById("hiringResultsBar");
        if (!bar) return;

        var isFiltered = activeAudience !== "fresher" || activeGroup !== "all" || !!activeQuery.trim();
        bar.innerHTML = [
            "<span class=\"hiring-result-pill\"><i class=\"fas fa-filter\"></i> Showing <strong>" + shown + "</strong> of " + total + " companies</span>",
            isFiltered ? "<button type=\"button\" class=\"hiring-reset-btn\" id=\"hiringResetBtn\"><i class=\"fas fa-rotate-left\"></i> Reset Filters</button>" : ""
        ].join("");

        var resetBtn = document.getElementById("hiringResetBtn");
        if (resetBtn) {
            resetBtn.addEventListener("click", function () {
                activeAudience = "fresher";
                activeGroup = "all";
                activeQuery = "";

                var audienceFresh = document.querySelector("#audienceTabs [data-audience='fresher']");
                var audienceButtons = document.querySelectorAll("#audienceTabs [data-audience]");
                audienceButtons.forEach(function (b) {
                    b.classList.remove("active");
                    b.setAttribute("aria-pressed", String(b === audienceFresh));
                });
                if (audienceFresh) audienceFresh.classList.add("active");

                var groupAll = document.querySelector("#companyGroupFilter [data-group='all']");
                var groupButtons = document.querySelectorAll("#companyGroupFilter [data-group]");
                groupButtons.forEach(function (b) {
                    b.classList.remove("active");
                    b.setAttribute("aria-pressed", String(b === groupAll));
                });
                if (groupAll) groupAll.classList.add("active");

                var search = document.getElementById("companySearch");
                if (search) search.value = "";

                renderCards();
                document.getElementById("companySearch").focus();
            });
        }
    }

    function renderCards() {
        var grid = document.getElementById("companyHiringGrid");
        if (!grid) return;

        var total = companyData.length;

        var filtered = companyData.filter(function (company) {
            var groupMatch = activeGroup === "all" || company.group === activeGroup;
            var q = activeQuery.trim().toLowerCase();
            var queryMatch = !q || company.name.toLowerCase().indexOf(q) >= 0 || company.aliases.join(" ").toLowerCase().indexOf(q) >= 0;
            return groupMatch && queryMatch;
        }).sort(function (a, b) {
            return a.name.localeCompare(b.name);
        });

        updateResultsBar(total, filtered.length);

        if (!filtered.length) {
            grid.innerHTML = "<div class=\"hiring-empty\"><h3>No matching company found</h3><p>Try a different search term or switch company group filters.</p></div>";
            return;
        }

        grid.innerHTML = filtered.map(function (company) {
            var guide = company.guides[activeAudience];
            var application = company.application[activeAudience];
            var pitfalls = company.pitfalls[activeAudience];
            var audienceProcess = getAudienceProcess(company);
            var audienceRounds = getAudienceRounds(company);

            var roundsHtml = audienceRounds.map(function (round) {
                return [
                    "<article class=\"hiring-round-item\">",
                    "<h4><i class=\"fas fa-compass\"></i> " + escapeHtml(round.name) + "</h4>",
                    "<p><strong>Preparation focus (confirm for role):</strong> " + escapeHtml(round.check) + "</p>",
                    "<p><strong>Practice approach:</strong> " + escapeHtml(round.clear) + "</p>",
                    "</article>"
                ].join("");
            }).join("");

            var processHtml = audienceProcess.map(function (step, idx) {
                return [
                    "<li>",
                    "<span class=\"hiring-step-badge\">" + (idx + 1) + "</span>",
                    "<span>" + escapeHtml(step) + "</span>",
                    "</li>"
                ].join("");
            }).join("");

            var audienceLabel = activeAudience === "fresher" ? "Fresher" : "Experienced";
            var groupName = groupLabels[company.group] || "General";
            var companyCode = getCompanyCode(company.name);
            var logoUrl = getCompanyLogoUrl(company.name);
            var focusHighlights = guide.focus.slice(0, 2);
            var snapshotBits = [
                "<span><i class=\"fas fa-route\"></i> " + audienceProcess.length + " example stages</span>",
                "<span><i class=\"fas fa-compass-drafting\"></i> " + audienceRounds.length + " preparation areas</span>",
                "<span><i class=\"fas fa-calendar-check\"></i> Suggested 60-day study plan</span>"
            ].join("");
            var logoHtml = logoUrl
                ? "<img src=\"" + escapeHtml(logoUrl) + "\" alt=\"" + escapeHtml(company.name) + " logo\" loading=\"lazy\" onerror=\"this.style.display='none';this.parentElement.classList.add('is-fallback');this.parentElement.textContent='" + escapeHtml(companyCode) + "';\">"
                : "";

            var disclosureHtml = [
                buildDisclosure(
                    "Scope, Role Differences, and Glossary",
                    "fa-circle-info",
                    buildPreparationContext(activeAudience),
                    true
                ),
                buildDisclosure(
                    "Illustrative Hiring Stages (Not Guaranteed)",
                    "fa-list-ol",
                    "<ol class=\"hiring-process-lane\">" + processHtml + "</ol>",
                    false
                ),
                buildDisclosure(
                    "Round-by-Round Playbook",
                    "fa-compass",
                    "<div class=\"hiring-round-grid\">" + roundsHtml + "</div>",
                    false
                ),
                buildDisclosure(
                    "Preparation Priorities for " + audienceLabel,
                    "fa-bullseye",
                    buildList(guide.focus),
                    false
                ),
                buildDisclosure(
                    "Application Strategy",
                    "fa-file-circle-check",
                    buildList(application),
                    false
                ),
                buildDisclosure(
                    "30/60 Day Execution Plan",
                    "fa-calendar-days",
                    [
                        "<div class=\"hiring-two-col\">",
                        "<div>",
                        "<h4>First 30 Days</h4>",
                        buildStudyList(guide.day30),
                        "</div>",
                        "<div>",
                        "<h4>Next 30 Days</h4>",
                        buildStudyList(guide.day60),
                        "</div>",
                        "</div>"
                    ].join(""),
                    false
                ),
                buildDisclosure(
                    "Behavioral Answer Structure and Practice",
                    "fa-comments",
                    buildBehavioralPractice(activeAudience),
                    false
                ),
                buildDisclosure(
                    "Practice Pitfalls (Not Official Rejection Criteria)",
                    "fa-triangle-exclamation",
                    "<section class=\"hiring-risk-block\">" + buildList(pitfalls) + "</section>",
                    false
                )
            ].join("");

            return [
                "<article class=\"hiring-card\">",
                "  <div class=\"hiring-card-head\">",
                "    <div class=\"hiring-brand\">",
                "      <span class=\"hiring-logo\" aria-hidden=\"true\">" + logoHtml + "</span>",
                "      <div>",
                "        <h3>" + escapeHtml(company.name) + "</h3>",
                "        <p class=\"hiring-card-tag\">" + escapeHtml(company.tag) + "</p>",
                "      </div>",
                "    </div>",
                "    <span class=\"hiring-updated\">Scope: " + escapeHtml(company.lastUpdated) + "</span>",
                "  </div>",
                "  <div class=\"hiring-card-topline\">",
                "    <span class=\"hiring-chip\"><i class=\"fas fa-building\"></i> " + escapeHtml(groupName) + "</span>",
                "    <span class=\"hiring-chip\"><i class=\"fas fa-user-tag\"></i> " + audienceLabel + " Strategy</span>",
                "    <span class=\"hiring-chip\"><i class=\"fas fa-sitemap\"></i> " + audienceRounds.length + " Practice Areas</span>",
                "  </div>",
                "  <div class=\"hiring-card-body\">",
                "    <div class=\"hiring-focus-strip\">",
                focusHighlights.map(function (item) { return "<p><i class=\"fas fa-check-circle\"></i> " + escapeHtml(item) + "</p>"; }).join(""),
                "    </div>",
                "    <div class=\"hiring-snapshot\">" + snapshotBits + "</div>",
                "    <div class=\"hiring-disclosure-stack\">",
                disclosureHtml,
                "    </div>",
                "  </div>",
                "</article>"
            ].join("");
        }).join("");
        enhanceExternalLinks(grid);
    }

    function setupAudienceTabs() {
        var tabs = document.querySelectorAll("#audienceTabs [data-audience]");
        tabs.forEach(function (tab) {
            tab.addEventListener("click", function () {
                activeAudience = tab.getAttribute("data-audience") || "fresher";
                tabs.forEach(function (b) {
                    b.classList.remove("active");
                    b.setAttribute("aria-pressed", String(b === tab));
                });
                tab.classList.add("active");
                renderCards();
            });
        });
    }

    function setupGroupFilter() {
        var buttons = document.querySelectorAll("#companyGroupFilter [data-group]");
        buttons.forEach(function (btn) {
            btn.addEventListener("click", function () {
                activeGroup = btn.getAttribute("data-group") || "all";
                buttons.forEach(function (b) {
                    b.classList.remove("active");
                    b.setAttribute("aria-pressed", String(b === btn));
                });
                btn.classList.add("active");
                renderCards();
            });
        });
    }

    function setupSearch() {
        var search = document.getElementById("companySearch");
        if (!search) return;
        search.addEventListener("input", function () {
            activeQuery = search.value || "";
            renderCards();
        });
    }

    document.addEventListener("DOMContentLoaded", function () {
        if (!document.querySelector("[data-page='hiring-procedures']")) return;
        setupAudienceTabs();
        setupGroupFilter();
        setupSearch();
        renderCards();
    });
})();
