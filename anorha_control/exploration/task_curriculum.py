"""
Task Curriculum - Structured training tasks for TRM precision training.
Includes aim trainers, typing tests, form completion, and long-horizon objectives.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import random


class TaskCategory(Enum):
    """Categories of training tasks."""
    PRECISION = "precision"      # Aim trainers, clicking accuracy
    TYPING = "typing"            # Speed typing, text entry
    FORMS = "forms"              # Form filling, data entry
    NAVIGATION = "navigation"    # Site exploration, search
    ECOMMERCE = "ecommerce"      # Shopping flows, checkout
    LONGHORIZON = "longhorizon"  # Multi-step complex tasks
    REALWORLD = "realworld"      # Real sites like Google, Airbnb, etc.


class Difficulty(Enum):
    """Task difficulty levels."""
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4


@dataclass
class Task:
    """A single training task."""
    name: str
    category: TaskCategory
    difficulty: Difficulty
    site: str
    objective: str
    success_hints: List[str]  # Visual hints for success (for GLM/VLM to verify)
    max_steps: int = 20
    sample_data: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return f"[{self.category.value}] {self.name} ({self.difficulty.name})"


# =============================================================================
# PRECISION TASKS - Aim Trainers (Critical for TRM accuracy)
# =============================================================================

PRECISION_TASKS = [
    # Aim Trainers (CRITICAL for mouse precision)
    Task(
        name="Human Benchmark - Aim Trainer",
        category=TaskCategory.PRECISION,
        difficulty=Difficulty.EASY,
        site="https://humanbenchmark.com/tests/aim",
        objective="Click 30 targets as fast as possible",
        success_hints=["Target clicked", "Remaining count decreased", "Time displayed"],
        max_steps=35,
    ),
    Task(
        name="Human Benchmark - Reaction Time",
        category=TaskCategory.PRECISION,
        difficulty=Difficulty.EASY,
        site="https://humanbenchmark.com/tests/reactiontime",
        objective="Click when the screen turns green",
        success_hints=["Green screen appeared", "Click time displayed", "ms result"],
        max_steps=10,
    ),
    
    # UI Interaction Precision
    Task(
        name="UI Playground - Dynamic Buttons",
        category=TaskCategory.PRECISION,
        difficulty=Difficulty.MEDIUM,
        site="http://uitestingplayground.com/click",
        objective="Click the button that moves and changes",
        success_hints=["Button clicked", "Button color changed", "Success message"],
        max_steps=15,
    ),
    Task(
        name="UI Playground - Hidden Layers",
        category=TaskCategory.PRECISION,
        difficulty=Difficulty.HARD,
        site="http://uitestingplayground.com/overlapped",
        objective="Click elements obscured by overlapping layers",
        success_hints=["Element revealed", "Click registered", "Scroll performed"],
        max_steps=10,
    ),
    Task(
        name="Locator Game",
        category=TaskCategory.PRECISION,
        difficulty=Difficulty.MEDIUM,
        site="https://testsmith-io.github.io/locator-game/",
        objective="Find and click the correct element based on hints",
        success_hints=["Correct!", "Level complete", "Score up"],
        max_steps=20,
    ),
    
    # Calculator (precision + cognitive)
    Task(
        name="Calculator - Basic Math",
        category=TaskCategory.PRECISION,
        difficulty=Difficulty.EASY,
        site="https://testsheepnz.github.io/BasicCalculator.html",
        objective="Solve: 42 + 13 = ?",
        success_hints=["55", "Result displayed", "Answer shown"],
        max_steps=8,
        sample_data={"expression": "42 + 13", "answer": "55"},
    ),
    Task(
        name="Calculator - Complex",
        category=TaskCategory.PRECISION,
        difficulty=Difficulty.MEDIUM,
        site="https://testsheepnz.github.io/BasicCalculator.html",
        objective="Solve: (25 * 4) - 17 = ?",
        success_hints=["83", "Result displayed"],
        max_steps=12,
        sample_data={"expression": "(25 * 4) - 17", "answer": "83"},
    ),
]


# =============================================================================
# TYPING TASKS - Speed and accuracy (Sandbox-compatible only)
# NOTE: Real-time typing sites (TypeRacer, MonkeyType, 10FastFingers) have
#       anti-bot detection and require real keyboard timing - they don't work
#       reliably with Playwright's simulated typing. Use form-based typing instead.
# =============================================================================

TYPING_TASKS = [
    # Form-based typing (works with Playwright)
    Task(
        name="DemoQA - Text Box Entry",
        category=TaskCategory.TYPING,
        difficulty=Difficulty.EASY,
        site="https://demoqa.com/text-box",
        objective="Fill in name, email, current address, and permanent address",
        success_hints=["Form filled", "Output displayed", "Data shown"],
        max_steps=15,
        sample_data={"name": "John Doe", "email": "john@example.com", "address": "123 Main St"},
    ),
    Task(
        name="The Internet - Text Input",
        category=TaskCategory.TYPING,
        difficulty=Difficulty.EASY,
        site="http://the-internet.herokuapp.com/inputs",
        objective="Type numbers into the input field",
        success_hints=["Number entered", "Input updated"],
        max_steps=8,
    ),
    Task(
        name="UI Playground - Text Input",
        category=TaskCategory.TYPING,
        difficulty=Difficulty.EASY,
        site="http://uitestingplayground.com/textinput",
        objective="Type a new button name and click the button to change its text",
        success_hints=["Button text changed", "Input accepted"],
        max_steps=8,
        sample_data={"button_name": "Anorha Rocks"},
    ),
    Task(
        name="Expand Testing - Textarea",
        category=TaskCategory.TYPING,
        difficulty=Difficulty.MEDIUM,
        site="https://practice.expandtesting.com/inputs",
        objective="Fill out various input types including textarea",
        success_hints=["Text entered", "Form filled"],
        max_steps=15,
    ),
    Task(
        name="W3Schools - Input Form",
        category=TaskCategory.TYPING,
        difficulty=Difficulty.EASY,
        site="https://www.w3schools.com/html/tryit.asp?filename=tryhtml_form_submit",
        objective="Type into the name and submit the form in the iframe",
        success_hints=["Form submitted", "Input accepted"],
        max_steps=10,
    ),
]



# =============================================================================
# FORM TASKS - Data entry and completion
# =============================================================================

# Sample data for form filling
SAMPLE_PEOPLE = [
    {"first": "John", "last": "Doe", "email": "john.doe@example.com", "phone": "555-123-4567", "address": "123 Main St", "city": "New York", "zip": "10001"},
    {"first": "Jane", "last": "Smith", "email": "jane.smith@test.com", "phone": "555-987-6543", "address": "456 Oak Ave", "city": "Los Angeles", "zip": "90001"},
    {"first": "Bob", "last": "Johnson", "email": "bob.j@demo.com", "phone": "555-456-7890", "address": "789 Pine Rd", "city": "Chicago", "zip": "60601"},
    {"first": "Alice", "last": "Williams", "email": "alice.w@sample.com", "phone": "555-321-9876", "address": "321 Elm St", "city": "Houston", "zip": "77001"},
]

FORM_TASKS = [
    Task(
        name="DemoQA - Practice Form",
        category=TaskCategory.FORMS,
        difficulty=Difficulty.EASY,
        site="https://demoqa.com/automation-practice-form",
        objective="Fill out the complete registration form and submit",
        success_hints=["Form submitted", "Success modal", "Thanks for submitting"],
        max_steps=25,
        sample_data=random.choice(SAMPLE_PEOPLE),
    ),
    Task(
        name="Sauce Demo - Login",
        category=TaskCategory.FORMS,
        difficulty=Difficulty.EASY,
        site="https://www.saucedemo.com/",
        objective="Login with username 'standard_user' and password 'secret_sauce'",
        success_hints=["Logged in", "Products page", "Inventory displayed"],
        max_steps=6,
        sample_data={"username": "standard_user", "password": "secret_sauce"},
    ),
    Task(
        name="OrangeHRM - Login",
        category=TaskCategory.FORMS,
        difficulty=Difficulty.EASY,
        site="https://opensource-demo.orangehrmlive.com/",
        objective="Login with Admin/admin123",
        success_hints=["Dashboard", "Logged in", "Welcome"],
        max_steps=6,
        sample_data={"username": "Admin", "password": "admin123"},
    ),
    Task(
        name="Contact List - Create Contact",
        category=TaskCategory.FORMS,
        difficulty=Difficulty.MEDIUM,
        site="https://thinking-tester-contact-list.herokuapp.com/",
        objective="Sign up and create a new contact",
        success_hints=["Contact added", "Contact list", "Name displayed"],
        max_steps=20,
        sample_data=random.choice(SAMPLE_PEOPLE),
    ),
    Task(
        name="Expand Testing - Forms",
        category=TaskCategory.FORMS,
        difficulty=Difficulty.MEDIUM,
        site="https://practice.expandtesting.com/",
        objective="Complete various form types",
        success_hints=["Form submitted", "Success", "Completed"],
        max_steps=15,
        sample_data=random.choice(SAMPLE_PEOPLE),
    ),
]


# =============================================================================
# NAVIGATION TASKS - Exploration and search
# =============================================================================

NAVIGATION_TASKS = [
    Task(
        name="Wikipedia - Find Article",
        category=TaskCategory.NAVIGATION,
        difficulty=Difficulty.EASY,
        site="https://en.wikipedia.org/wiki/Main_Page",
        objective="Navigate to the article about 'Machine Learning' by clicking links",
        success_hints=["Machine Learning", "Article found", "ML page displayed"],
        max_steps=15,
    ),
    Task(
        name="Wikipedia - Deep Dive",
        category=TaskCategory.NAVIGATION,
        difficulty=Difficulty.HARD,
        site="https://en.wikipedia.org/wiki/Special:Random",
        objective="Starting from a random article, reach the 'Philosophy' article by clicking links",
        success_hints=["Philosophy", "Article reached", "Navigation complete"],
        max_steps=30,
    ),
    Task(
        name="The Internet - Page Navigation",
        category=TaskCategory.NAVIGATION,
        difficulty=Difficulty.EASY,
        site="http://the-internet.herokuapp.com/",
        objective="Navigate to 'Form Authentication' and login with tomsmith/SuperSecretPassword!",
        success_hints=["Logged in", "Secure Area", "Welcome"],
        max_steps=10,
        sample_data={"username": "tomsmith", "password": "SuperSecretPassword!"},
    ),
    Task(
        name="GitHub Users Search",
        category=TaskCategory.NAVIGATION,
        difficulty=Difficulty.MEDIUM,
        site="https://gh-users-search.netlify.app/",
        objective="Search for 'torvalds' and view their profile",
        success_hints=["Linus Torvalds", "Profile displayed", "Followers shown"],
        max_steps=10,
        sample_data={"query": "torvalds"},
    ),
]


# =============================================================================
# E-COMMERCE TASKS - Shopping flows
# =============================================================================

ECOMMERCE_TASKS = [
    Task(
        name="Sauce Demo - Full Checkout",
        category=TaskCategory.ECOMMERCE,
        difficulty=Difficulty.MEDIUM,
        site="https://www.saucedemo.com/",
        objective="Login, add 2 items to cart, complete checkout",
        success_hints=["Order complete", "Thank you", "Checkout complete"],
        max_steps=20,
        sample_data={"username": "standard_user", "password": "secret_sauce", **random.choice(SAMPLE_PEOPLE)},
    ),
    Task(
        name="Coffee Cart - Order Coffee",
        category=TaskCategory.ECOMMERCE,
        difficulty=Difficulty.EASY,
        site="https://coffee-cart.app/",
        objective="Add a cappuccino and espresso to cart, complete order",
        success_hints=["Cart updated", "Checkout", "Order placed"],
        max_steps=12,
    ),
    Task(
        name="BookCart - Buy Book",
        category=TaskCategory.ECOMMERCE,
        difficulty=Difficulty.MEDIUM,
        site="https://bookcart.azurewebsites.net/",
        objective="Search for a book, add to cart, proceed to checkout",
        success_hints=["Added to cart", "Cart updated", "Checkout page"],
        max_steps=15,
    ),
    Task(
        name="Practice Software Testing - Checkout",
        category=TaskCategory.ECOMMERCE,
        difficulty=Difficulty.HARD,
        site="https://practicesoftwaretesting.com/",
        objective="Complete a full purchase with registration",
        success_hints=["Order confirmed", "Purchase complete", "Thank you"],
        max_steps=30,
        sample_data=random.choice(SAMPLE_PEOPLE),
    ),
]


# =============================================================================
# LONG-HORIZON TASKS - Complex multi-step objectives
# =============================================================================

LONGHORIZON_TASKS = [
    Task(
        name="Research Task - Find Answer",
        category=TaskCategory.LONGHORIZON,
        difficulty=Difficulty.HARD,
        site="https://www.google.com/",
        objective="Search for 'What year was Python created?' and find the answer",
        success_hints=["1991", "Guido van Rossum", "Answer found"],
        max_steps=15,
        sample_data={"answer": "1991"},
    ),
    Task(
        name="Wikipedia Research",
        category=TaskCategory.LONGHORIZON,
        difficulty=Difficulty.HARD,
        site="https://en.wikipedia.org/",
        objective="Find the population of Tokyo from the Wikipedia article",
        success_hints=["13 million", "14 million", "population", "Tokyo"],
        max_steps=20,
    ),
    Task(
        name="Multi-Form Registration",
        category=TaskCategory.LONGHORIZON,
        difficulty=Difficulty.EXPERT,
        site="https://practicesoftwaretesting.com/",
        objective="Complete full user registration, login, and profile setup",
        success_hints=["Profile complete", "Account created", "Logged in"],
        max_steps=40,
        sample_data=random.choice(SAMPLE_PEOPLE),
    ),
]


# =============================================================================
# REAL-WORLD TASKS - Complex real-world scenarios
# =============================================================================

REALWORLD_TASKS = [
    Task(
        name="Google Flights - Search Flights",
        category=TaskCategory.REALWORLD,
        difficulty=Difficulty.HARD,
        site="https://www.google.com/travel/flights",
        objective="Search for round-trip flights from San Francisco to Tokyo for next month",
        success_hints=["Flight results", "Prices shown", "Departure", "SFO", "TYO"],
        max_steps=20,
        sample_data={
            "from": "San Francisco",
            "to": "Tokyo",
            "departure_date": "next month",
            "trip_type": "round-trip",
        },
    ),
    Task(
        name="Airbnb - Browse Listings",
        category=TaskCategory.REALWORLD,
        difficulty=Difficulty.HARD,
        site="https://www.airbnb.com/",
        objective="Search for a place to stay in Paris for 2 guests, browse listings",
        success_hints=["Listings", "Paris", "guests", "Price", "Results"],
        max_steps=25,
        sample_data={
            "location": "Paris, France",
            "guests": "2",
            "check_in": "next month",
        },
    ),
    Task(
        name="LinkedIn - Job Search (Public)",
        category=TaskCategory.REALWORLD,
        difficulty=Difficulty.HARD,
        site="https://www.linkedin.com/jobs/search/",
        objective="Search for 'Software Engineer' jobs in 'Remote' location",
        success_hints=["Job listings", "Software Engineer", "Remote", "Apply"],
        max_steps=20,
        sample_data={
            "job_title": "Software Engineer",
            "location": "Remote",
        },
    ),
    Task(
        name="Product Hunt - Browse Today",
        category=TaskCategory.REALWORLD,
        difficulty=Difficulty.MEDIUM,
        site="https://www.producthunt.com/",
        objective="Browse today's top products and click on the #1 product",
        success_hints=["Product page", "Upvotes", "Comments", "Details"],
        max_steps=15,
    ),
    Task(
        name="GitHub - Search Repository",
        category=TaskCategory.REALWORLD,
        difficulty=Difficulty.MEDIUM,
        site="https://github.com/search",
        objective="Search for 'machine learning' repositories and view the top result",
        success_hints=["Repository", "Stars", "README", "Code"],
        max_steps=15,
        sample_data={"query": "machine learning"},
    ),
    Task(
        name="Booking.com - Hotel Search",
        category=TaskCategory.REALWORLD,
        difficulty=Difficulty.HARD,
        site="https://www.booking.com/",
        objective="Search for hotels in Barcelona, Spain for 2 adults",
        success_hints=["Hotels", "Barcelona", "Price", "Availability", "Rating"],
        max_steps=25,
        sample_data={
            "location": "Barcelona, Spain",
            "guests": "2 adults",
        },
    ),
]


# =============================================================================
# CURRICULUM CLASS
# =============================================================================

class TaskCurriculum:
    """
    Manages the training curriculum with progressive difficulty.
    
    Training Estimates (for TRM with 2.3M params):
    - Easy tasks: ~500 successful episodes each
    - Medium tasks: ~1,000 successful episodes each
    - Hard tasks: ~2,000 successful episodes each
    - Total for basic proficiency: ~10,000-20,000 experiences
    - Total for high accuracy: ~50,000-100,000 experiences
    
    At 10 steps/episode, 1 episode/minute:
    - Basic proficiency: ~17-33 hours of exploration
    - High accuracy: ~83-166 hours of exploration
    """
    
    # All tasks organized by category
    ALL_TASKS = {
        TaskCategory.PRECISION: PRECISION_TASKS,
        TaskCategory.TYPING: TYPING_TASKS,
        TaskCategory.FORMS: FORM_TASKS,
        TaskCategory.NAVIGATION: NAVIGATION_TASKS,
        TaskCategory.ECOMMERCE: ECOMMERCE_TASKS,
        TaskCategory.LONGHORIZON: LONGHORIZON_TASKS,
        TaskCategory.REALWORLD: REALWORLD_TASKS,
    }
    
    # Category weights for balanced training
    CATEGORY_WEIGHTS = {
        TaskCategory.PRECISION: 0.35,   # Most important for TRM accuracy
        TaskCategory.TYPING: 0.15,
        TaskCategory.FORMS: 0.15,
        TaskCategory.NAVIGATION: 0.10,
        TaskCategory.ECOMMERCE: 0.05,
        TaskCategory.LONGHORIZON: 0.05,
        TaskCategory.REALWORLD: 0.15,   # High weight for real-world application
    }
    
    def __init__(self, max_difficulty: Difficulty = Difficulty.MEDIUM):
        self.max_difficulty = max_difficulty
        self.completed_tasks: Dict[str, int] = {}  # task_name -> success_count
        self.current_task: Optional[Task] = None
    
    def get_training_estimate(self) -> Dict[str, Any]:
        """Estimate training time required."""
        return {
            "basic_proficiency": {
                "experiences": "10,000-20,000",
                "episodes": "1,000-2,000",
                "time_hours": "17-33",
            },
            "high_accuracy": {
                "experiences": "50,000-100,000",
                "episodes": "5,000-10,000",
                "time_hours": "83-166",
            },
            "current_recommendation": "Start with precision tasks (aim trainers) for 5 hours, then mix in forms and navigation"
        }
    
    def sample_task(self, category: TaskCategory = None) -> Task:
        """Sample a task, optionally from a specific category."""
        if category:
            tasks = self.ALL_TASKS[category]
        else:
            # Weighted category selection
            categories = list(self.CATEGORY_WEIGHTS.keys())
            weights = list(self.CATEGORY_WEIGHTS.values())
            category = random.choices(categories, weights=weights)[0]
            tasks = self.ALL_TASKS[category]
        
        # Filter by difficulty
        valid_tasks = [t for t in tasks if t.difficulty.value <= self.max_difficulty.value]
        
        if not valid_tasks:
            valid_tasks = tasks  # Fallback to all tasks in category
        
        task = random.choice(valid_tasks)
        
        # Refresh sample data for variety
        if task.sample_data and "first" in task.sample_data:
            task.sample_data = random.choice(SAMPLE_PEOPLE)
        
        self.current_task = task
        return task
    
    def sample_precision_task(self) -> Task:
        """Sample specifically from precision/aim trainer tasks."""
        return self.sample_task(TaskCategory.PRECISION)
    
    def sample_typing_task(self) -> Task:
        """Sample from typing tasks."""
        return self.sample_task(TaskCategory.TYPING)
    
    def mark_success(self, task: Task):
        """Record a successful task completion."""
        if task.name not in self.completed_tasks:
            self.completed_tasks[task.name] = 0
        self.completed_tasks[task.name] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get curriculum progress stats."""
        total_successes = sum(self.completed_tasks.values())
        by_category = {}
        
        for category, tasks in self.ALL_TASKS.items():
            category_successes = sum(
                self.completed_tasks.get(t.name, 0) for t in tasks
            )
            by_category[category.value] = category_successes
        
        return {
            "total_successes": total_successes,
            "by_category": by_category,
            "unique_tasks_completed": len(self.completed_tasks),
            "total_task_types": sum(len(tasks) for tasks in self.ALL_TASKS.values()),
        }
    
    def increase_difficulty(self):
        """Progress to harder tasks."""
        if self.max_difficulty == Difficulty.EASY:
            self.max_difficulty = Difficulty.MEDIUM
        elif self.max_difficulty == Difficulty.MEDIUM:
            self.max_difficulty = Difficulty.HARD
        elif self.max_difficulty == Difficulty.HARD:
            self.max_difficulty = Difficulty.EXPERT
        print(f"[Curriculum] Difficulty increased to {self.max_difficulty.name}")
    
    def get_all_sites(self) -> List[str]:
        """Get all unique sites in the curriculum."""
        sites = set()
        for tasks in self.ALL_TASKS.values():
            for task in tasks:
                sites.add(task.site)
        return list(sites)


# CLI for testing
if __name__ == "__main__":
    curriculum = TaskCurriculum()
    
    print("=== Task Curriculum ===\n")
    print("Training Estimates:")
    for key, value in curriculum.get_training_estimate().items():
        print(f"  {key}: {value}")
    
    print(f"\nTotal sites: {len(curriculum.get_all_sites())}")
    print(f"Total task types: {sum(len(tasks) for tasks in curriculum.ALL_TASKS.values())}")
    
    print("\n=== Sample Tasks ===")
    for category in TaskCategory:
        task = curriculum.sample_task(category)
        print(f"\n{task}")
        print(f"  Site: {task.site}")
        print(f"  Objective: {task.objective}")
