(function () {
    var companyData = [
        {
            name: "Microsoft",
            group: "product",
            tag: "Cloud + Product Engineering",
            lastUpdated: "Jul 2026",
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
                    day30: ["DSA fundamentals + 60 timed problems", "Revise CS core: OS, DBMS, Networking basics", "Mock interview every weekend"],
                    day60: ["System design fundamentals for interns/new grads", "Refine project storytelling", "Run 6 full interview simulations"]
                },
                experienced: {
                    focus: [
                        "Strengthen LLD/HLD communication and trade-off reasoning.",
                        "Be ready to discuss production architecture and reliability decisions.",
                        "Demonstrate leadership signals even as IC: mentorship, influence, ownership."
                    ],
                    day30: ["Rebuild design basics + coding speed refresh", "Prepare 5 deep project stories", "Behavioral story bank for leadership principles"],
                    day60: ["Advanced mock loops with design + coding combo", "Negotiate role-level alignment", "Targeted revision using rejected-round feedback"]
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
            lastUpdated: "Jul 2026",
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
                    day30: ["Daily OA drills", "Top LP story prep", "Two mock coding rounds weekly"],
                    day60: ["System design lite for new grads", "Behavioral drills", "Final mixed interview simulation"]
                },
                experienced: {
                    focus: ["Design depth and scale decisions", "LP depth with conflicts/trade-offs", "Production troubleshooting narratives"],
                    day30: ["Coding refresh + design templates", "Prepare impact metrics portfolio", "Leadership story rehearsals"],
                    day60: ["Bar-raiser style mocks", "Cross-functional stakeholder scenarios", "Offer-stage strategy prep"]
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
            lastUpdated: "Jul 2026",
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
                    day30: ["Solve 2 problems/day with spoken reasoning", "Weekly peer mock", "Build one resume-ready project"],
                    day60: ["Design basics and API thinking", "Behavioral narrative polish", "6 end-to-end mock interviews"]
                },
                experienced: {
                    focus: ["High signal design communication", "Code quality and correctness under probing", "Cross-team influence examples"],
                    day30: ["Coding and design refresh", "Document architecture narratives", "Behavioral and leadership story bank"],
                    day60: ["Panel-style mock loops", "Feedback-driven gap correction", "Team matching prep"]
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
            lastUpdated: "Jul 2026",
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
                    day30: ["Aptitude drills", "Role-specific technical revision", "Mock HR and managerial rounds"],
                    day60: ["Client-case simulation", "Resume refinement", "Cross-question readiness on projects"]
                },
                experienced: {
                    focus: ["Consulting-style structured answers", "Delivery metrics and escalation handling", "Leadership in execution"],
                    day30: ["Rebuild story bank around projects", "Strengthen domain tech gaps", "Behavioral mock loops"],
                    day60: ["Client simulation rounds", "Offer-stage role fit discussion", "Feedback-based refinements"]
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
            lastUpdated: "Jul 2026",
            aliases: ["accenture"],
            process: ["Application", "Cognitive + coding assessments", "Technical interview", "Managerial/HR rounds", "Offer"],
            rounds: [
                { name: "Assessments", check: "Aptitude, communication, coding basics", clear: "Practice full test format and strict timing." },
                { name: "Technical", check: "Project implementation and stack fundamentals", clear: "Use clear architecture and module ownership explanation." },
                { name: "HR", check: "Adaptability and culture fit", clear: "Answer with examples of learning and teamwork." }
            ],
            application: {
                fresher: ["Prepare for adaptive test patterns and sectional cutoffs.", "Build one cloud/data/automation mini-project with presentation-ready summary.", "Strengthen spoken explanation and confidence."],
                experienced: ["Map profile to role family and domain vertical.", "Prepare migration, modernization, and delivery stories.", "Show process maturity: quality, risk, and stakeholder communication."]
            },
            guides: {
                fresher: {
                    focus: ["Assessment readiness", "Technical fundamentals", "Communication polish"],
                    day30: ["Daily aptitude + coding", "Project summary rehearsals", "Mock HR sessions"],
                    day60: ["Advanced timed tests", "Role-based technical grilling", "Interview repetition and feedback"]
                },
                experienced: {
                    focus: ["Project-to-business impact mapping", "Leadership and collaboration examples", "Domain articulation"],
                    day30: ["Assessment warmup", "Prepare major delivery examples", "Managerial Q&A practice"],
                    day60: ["Scenario interviews", "Compensation and level readiness", "Targeted weak-area correction"]
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
            lastUpdated: "Jul 2026",
            aliases: ["infy"],
            process: ["Campus/off-campus application", "Aptitude + coding test", "Technical round", "HR round", "Offer and onboarding"],
            rounds: [
                { name: "Test", check: "Reasoning, quant, coding basics", clear: "Maintain sectional accuracy and avoid negative scoring traps." },
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
                    day30: ["Daily aptitude drills", "Revise OOP/DBMS basics", "Mock technical interview weekly"],
                    day60: ["Timed full assessments", "Role-aligned coding revision", "HR narrative refinement"]
                },
                experienced: {
                    focus: ["Delivery impact and reliability", "Client communication", "Upskilling evidence"],
                    day30: ["Resume metrics rewrite", "Stack revision by role", "Behavioral scenario prep"],
                    day60: ["Panel simulation", "Role transition positioning", "Negotiation and offer readiness"]
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
            lastUpdated: "Jul 2026",
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
                    day30: ["Section-wise NQT prep", "Coding basics revision", "Weekly technical mock"],
                    day60: ["Full test simulation", "Interview answer structure", "Final HR readiness"]
                },
                experienced: {
                    focus: ["Delivery discipline and reliability", "Client communication", "Modern stack adaptation"],
                    day30: ["Resume impact edits", "Domain-technology revision", "Scenario interview prep"],
                    day60: ["Project deep-dive rehearsals", "Cross-question handling", "Offer conversion strategy"]
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
            lastUpdated: "Jul 2026",
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
                    day30: ["Aptitude and coding plan", "CS fundamentals revision", "Interview speaking practice"],
                    day60: ["Mock rounds", "Project defense practice", "HR fit preparation"]
                },
                experienced: {
                    focus: ["Operational ownership", "Troubleshooting examples", "Delivery leadership"],
                    day30: ["Incident story preparation", "Tech stack refresh", "Behavioral Q&A"],
                    day60: ["Managerial mock loops", "Impact-based storytelling", "Offer readiness"]
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
            lastUpdated: "Jul 2026",
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
                    day30: ["Timed tests", "Technical basics revision", "Mock interviews"],
                    day60: ["Role-focused prep", "Behavioral story polish", "Final conversion prep"]
                },
                experienced: {
                    focus: ["Client-delivery evidence", "Technical depth for target role", "Cross-functional collaboration"],
                    day30: ["Resume and impact cleanup", "Role-stack revision", "Managerial interview prep"],
                    day60: ["Panel simulations", "Negotiation readiness", "Gap-closing sprint"]
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
            lastUpdated: "Jul 2026",
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
                    day30: ["Daily test prep", "Language and DB revision", "Mock interview practice"],
                    day60: ["Full simulation rounds", "Resume and project polish", "HR preparation"]
                },
                experienced: {
                    focus: ["Delivery impact", "Support and maintenance maturity", "Upskilling continuity"],
                    day30: ["Impact metric collection", "Technical revision", "Behavioral rehearsals"],
                    day60: ["Scenario rounds", "Offer-level positioning", "Gap closure"]
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
            lastUpdated: "Jul 2026",
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
                    day30: ["Assessment preparation", "Project walkthrough practice", "Mock technical rounds"],
                    day60: ["Full process simulation", "Behavioral polish", "Company-specific Q&A readiness"]
                },
                experienced: {
                    focus: ["Delivery metrics", "Role-depth evidence", "Stakeholder stories"],
                    day30: ["Resume impact update", "Tech revision", "Managerial prep"],
                    day60: ["Scenario drills", "Advanced interview mock", "Offer conversion plan"]
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
            lastUpdated: "Jul 2026",
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
                    day30: ["Coding basics and API practice", "SQL and DB focus", "Mock technical rounds"],
                    day60: ["Domain-oriented case discussions", "Behavioral readiness", "End-to-end interview simulation"]
                },
                experienced: {
                    focus: ["Enterprise architecture choices", "Business process mapping", "Delivery reliability"],
                    day30: ["Project stories with domain outcomes", "Design and coding refresh", "Managerial prep"],
                    day60: ["Panel simulations", "Product-fit interview drills", "Negotiation prep"]
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
            lastUpdated: "Jul 2026",
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
                    day30: ["Coding and OOP drills", "Integration mini-project", "Behavioral prep"],
                    day60: ["Design basics revision", "Platform interview practice", "Full mock loops"]
                },
                experienced: {
                    focus: ["Platform architecture depth", "Data/integration trade-offs", "Cross-team leadership"],
                    day30: ["Design and coding refresh", "Project impact stories", "Behavioral evidence prep"],
                    day60: ["Advanced panel mocks", "Product and scale discussions", "Offer conversion strategy"]
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
            lastUpdated: "Jul 2026",
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
                    day30: ["Core technical revision", "Business-case project prep", "Mock interviews"],
                    day60: ["Domain scenario practice", "Design basics", "Behavioral preparation"]
                },
                experienced: {
                    focus: ["Enterprise design depth", "Business-tech translation", "Program delivery ownership"],
                    day30: ["Project story cataloging", "Architecture refresh", "Stakeholder scenario prep"],
                    day60: ["Panel interview simulations", "Domain probing readiness", "Offer-stage alignment"]
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
            lastUpdated: "Jul 2026",
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
                    day30: ["Daily coding drills", "Cloud basics revision", "Project storytelling practice"],
                    day60: ["Design-lite prep", "Mock interviews", "Behavioral polish"]
                },
                experienced: {
                    focus: ["Architecture and modernization depth", "Cross-team execution", "Business impact articulation"],
                    day30: ["Technical and design refresh", "Project metrics preparation", "Behavioral drills"],
                    day60: ["Advanced interview simulation", "Targeted weak-area improvement", "Offer conversion planning"]
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
                audienceButtons.forEach(function (b) { b.classList.remove("active"); });
                if (audienceFresh) audienceFresh.classList.add("active");

                var groupAll = document.querySelector("#companyGroupFilter [data-group='all']");
                var groupButtons = document.querySelectorAll("#companyGroupFilter [data-group]");
                groupButtons.forEach(function (b) { b.classList.remove("active"); });
                if (groupAll) groupAll.classList.add("active");

                var search = document.getElementById("companySearch");
                if (search) search.value = "";

                renderCards();
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
                    "<p><strong>What they check:</strong> " + escapeHtml(round.check) + "</p>",
                    "<p><strong>How to clear:</strong> " + escapeHtml(round.clear) + "</p>",
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
                "<span><i class=\"fas fa-route\"></i> " + audienceProcess.length + " process steps</span>",
                "<span><i class=\"fas fa-compass-drafting\"></i> " + audienceRounds.length + " interview rounds</span>",
                "<span><i class=\"fas fa-calendar-check\"></i> 60-day execution plan</span>"
            ].join("");
            var logoHtml = logoUrl
                ? "<img src=\"" + escapeHtml(logoUrl) + "\" alt=\"" + escapeHtml(company.name) + " logo\" loading=\"lazy\" onerror=\"this.style.display='none';this.parentElement.classList.add('is-fallback');this.parentElement.textContent='" + escapeHtml(companyCode) + "';\">"
                : "";

            var disclosureHtml = [
                buildDisclosure(
                    "Typical Hiring Procedure",
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
                    "How to Clear as " + audienceLabel,
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
                        buildList(guide.day30),
                        "</div>",
                        "<div>",
                        "<h4>Next 30 Days</h4>",
                        buildList(guide.day60),
                        "</div>",
                        "</div>"
                    ].join(""),
                    false
                ),
                buildDisclosure(
                    "Common Rejection Reasons",
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
                "    <span class=\"hiring-updated\">Updated: " + escapeHtml(company.lastUpdated) + "</span>",
                "  </div>",
                "  <div class=\"hiring-card-topline\">",
                "    <span class=\"hiring-chip\"><i class=\"fas fa-building\"></i> " + escapeHtml(groupName) + "</span>",
                "    <span class=\"hiring-chip\"><i class=\"fas fa-user-tag\"></i> " + audienceLabel + " Strategy</span>",
                "    <span class=\"hiring-chip\"><i class=\"fas fa-sitemap\"></i> " + audienceRounds.length + " Key Rounds</span>",
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
    }

    function setupAudienceTabs() {
        var tabs = document.querySelectorAll("#audienceTabs [data-audience]");
        tabs.forEach(function (tab) {
            tab.addEventListener("click", function () {
                activeAudience = tab.getAttribute("data-audience") || "fresher";
                tabs.forEach(function (b) { b.classList.remove("active"); });
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
                buttons.forEach(function (b) { b.classList.remove("active"); });
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
