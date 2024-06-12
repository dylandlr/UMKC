import cohere
from cohere import ClassifyExample
import pandas as pd

# Initialize the Cohere client with your API key
co = cohere.Client('sWXDjjlDE2k2f2fcRdavQbNJ07q14aVp7DRAmVBN')  # Replace with your actual API key

# Inputs from the balanced dataset
inputs = [
    "Claim your lottery winnings now!", "Special discount on your next purchase", 
    "Your Subscription is Active", "You've received a new message", 
    "Your invoice is ready for payment", "Team Outing Details", 
    "Subscription Renewal Confirmation", "Immediate action required", 
    "Reminder: Payment overdue", "Exclusive access to premium content", 
    "Update your password to avoid suspension", "Get a free sample of our new product", 
    "Meeting Reminder: Tomorrow at 10 AM", "Receive your free credit score report", 
    "Annual Company Picnic Information", "Your Invoice is Ready", 
    "Account Verification Successful", "Feedback Request on Recent Purchase", 
    "Invitation to Alumni Reunion", "Holiday Greetings from Our Team", 
    "Limited time offer! Get 50% off on all products", "Final notice: Update your billing information", 
    "Thank you for your purchase", "Welcome to Our Community", 
    "Free trial offer - sign up now", "Win a free trip to the Bahamas", 
    "Invitation to Family Gathering", "Meeting Rescheduled for Next Week", 
    "Unlock the secrets to financial freedom", "Last chance to claim your prize", 
    "Notice of Privacy Policy Update", "Secure your account with this quick update", 
    "Congratulations on your new job offer", "Monthly Performance Report", 
    "Congratulations on Your Promotion", "Important: Your account has been suspended", 
    "Donation Receipt from Charity", "Special Offer Just for You", 
    "Customer Satisfaction Survey", "Exclusive Invite to Webinar", 
    "Weekly Recipe: Try Something New", "Your Order has been Shipped", 
    "You have a new friend request", "Training Session Reminder", 
    "Workshop Registration Confirmation", "Upcoming Webinar on Marketing", 
    "Confirm your subscription now", "Get rich quick with this simple trick", 
    "Verify your email address to continue", "Weekly Team Meeting Agenda", 
    "Important update regarding your account", "Join our affiliate program and earn money", 
    "Your order has been shipped", "Congratulations! You are our lucky winner", 
    "Limited stock available - buy now", "Update Your Contact Information", 
    "Your account has been compromised", "Password Reset Successful", 
    "Claim your free gift today", "New message from your bank", 
    "Your Application has been Approved", "Event Reminder: Charity Run", 
    "You're eligible for a cashback reward", "Project Update: Status Report", 
    "Your Package has been Delivered", "Billing Statement Available Online", 
    "Happy Birthday! Celebrate with Us", "Quarterly Business Review", 
    "Important notice regarding your health", "Get paid to take surveys online", 
    "Schedule Your Performance Review", "Congratulations! You've won a $1000 gift card!", 
    "Your Order is on Its Way", "Urgent: Update your account information now", 
    "Doctor Appointment Confirmation", "Membership Renewal Notification", 
    "Job Interview Scheduled for Friday", "Thank You for Your Purchase", 
    "Get a loan with low interest rates", "Workshop Materials Available Online", 
    "Your subscription is about to expire", "New Feature Announcement", 
    "You've been selected for a free vacation package", "Exclusive deal just for you - act fast!", 
    "New Message from HR", "Act now and receive a special bonus", 
    "Your package is out for delivery", "Your payment was successful", 
    "Monthly Newsletter - June Edition", "Protect your account with two-factor authentication"
]

# Define the examples for training the model
examples = [
    ClassifyExample(text="?? You Earned 500 GCLoot Points", label="not spam"), 
    ClassifyExample(text="?? Your GitHub launch code", label="not spam"), 
    ClassifyExample(text="[The Virtual Reward Center] Re: ** Clarifications", label="not spam"), 
    ClassifyExample(text="AFE Model Casting Call ", label="not spam"), 
    ClassifyExample(text="AFE Model Casting Call ", label="not spam"), 
    ClassifyExample(text="amazon.com.tr, action needed: Sign-in", label="not spam"), 
    ClassifyExample(text="Appen 9 Project Invite - A5655?Request detail 27?Asia Image Collection Project", label="not spam"), 
    ClassifyExample(text="APPLICATION PROCESS ", label="not spam"), 
    ClassifyExample(text="Are you a luxury traveller or know someone who is??", label="not spam"), 
    ClassifyExample(text="Celebrate 15 years of the Bible App!", label="not spam"), 
    ClassifyExample(text="Changes to our Terms and Conditions and Privacy Notice", label="not spam"), 
    ClassifyExample(text="Combating scams in the Discogs Marketplace", label="not spam"), 
    ClassifyExample(text="Come back and keep earning with Quick Pay Survey", label="not spam"), 
    ClassifyExample(text="Congratulations! You have been selected for a special scholarship from Unicaf ", label="not spam"), 
    ClassifyExample(text="Customer Service Officer + 19 new jobs - Job Alert from JobStreet.com", label="not spam"), 
    ClassifyExample(text="denis : Your Amazon Prime Membership Cancellation", label="not spam"), 
    ClassifyExample(text="Feedback Request: How are we doing?", label="not spam"), 
    ClassifyExample(text="Get ready for Test [Co-Deo]", label="not spam"), 
    ClassifyExample(text="Gut Health & Weight Loss Plan - Role of Supplement", label="not spam"), 
    ClassifyExample(text="HOW TO CANCEL UIF REGISTRATION ", label="not spam"), 
    ClassifyExample(text="If coding in C++ is so extremely hard, why do we have so many PlayStation and Nintend...?", label="not spam"), 
    ClassifyExample(text="June 2023 Account Statement ", label="not spam"), 
    ClassifyExample(text="Junior Admin Assistant ", label="not spam"), 
    ClassifyExample(text="Linton, take the next step on your Windows device by confirming your Google Account settings", label="not spam"), 
    ClassifyExample(text="?? the secrets to SUCCESS", label="spam"), 
    ClassifyExample(text="10-1 MLB Expert Inside, Plus Everything You Need To Have A BLOCKBUSTER Saturday", label="spam"), 
    ClassifyExample(text="Affordable American MBA degree ($180month)", label="spam"), 
    ClassifyExample(text="Combating scams in the Discogs Marketplace", label="spam"), 
    ClassifyExample(text="Do you have $7 Walid?", label="spam"), 
    ClassifyExample(text="English", label="spam"), 
    ClassifyExample(text="English", label="spam"), 
    ClassifyExample(text="English", label="spam"), 
    ClassifyExample(text="Flowing with the markets", label="spam"), 
    ClassifyExample(text="Get the Competitive Edge: ?? Project-Winning Strategies Revealed", label="spam"), 
    ClassifyExample(text="Get the Competitive Edge: ?? Project-Winning Strategies Revealed", label="spam"), 
    ClassifyExample(text="Hello lun New message from PhebeHotHoney", label="spam"), 
    ClassifyExample(text="Hi lun Message from TuPussycat", label="spam"), 
    ClassifyExample(text="Job Opportunities that pay ?600,000+ and up in Nigeria", label="spam"), 
    ClassifyExample(text="John, here’s a discount for your next safe rides ??", label="spam"), 
    ClassifyExample(text="Limited offer. Hurry up!", label="spam"), 
    ClassifyExample(text="Live has arrived on TIDAL ??", label="spam"), 
    ClassifyExample(text="Meeting with Walid TOMORROW", label="spam"), 
    ClassifyExample(text="SFI Weekly News (July 17, 2023)", label="spam"), 
    ClassifyExample(text="Style with Denim", label="spam"), 
    ClassifyExample(text="The Last Call!", label="spam"), 
    ClassifyExample(text="Time is of the Essence!", label="spam"), 
    ClassifyExample(text="try out our high-quality rides for less! ?", label="spam"), 
    ClassifyExample(text="Walid, Get Paid to Listen to Music ", label="spam"), 
    ClassifyExample(text="Yes- it is a numbers game", label="spam"), 
    ClassifyExample(text="Your Account Was Accessed From a New Device", label="spam")
    ]
# Get the response from the model
response = co.classify(
    model='embed-english-v2.0',
    inputs=inputs,
    examples=examples
)

# Extract the actual and predicted labels for comparison
predicted_labels = [classification.prediction for classification in response.classifications]

# Create a DataFrame to compare actual and predicted labels
comparison_df = pd.DataFrame({
    'title': inputs,
    'actual': ['spam'] * 45 + ['not spam'] * 45,
    'predicted': predicted_labels
})

# Save the comparison DataFrame to a CSV file
comparison_file_path = 'model_performance_comparison.csv'
comparison_df.to_csv(comparison_file_path, index=False)

print(f"Comparison DataFrame saved to {comparison_file_path}")
