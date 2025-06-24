import random
import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time
import backoff
import misc_utils as misc

CONFIG_DIR = "config"
API_KEY_FILE_NAME = "api_key.txt"
DATASET_OUTPUT_DIR = "call_transcript_dataset_output"
DATASET_OUTPUT_BASE_NAME = "generated_transcript_dataset_"

# Example prompts for each category with longer transcripts
EXAMPLE_PROMPTS = {
    "fraud": {
        "Emotional Manipulation": [
            "scammer\tmale\tNamaste, this is Ravi from Child Welfare NGO. We’re helping a sick child who needs surgery.\n"
            "person 1\tfemale\tOh, that’s heartbreaking. How can I help?\n"
            "scammer\tmale\tWe need ₹10,000 urgently. You can send it to http://childcare-india.org.\n"
            "person 1\tfemale\tThat’s a lot. Can you share more details about the child?\n"
            "scammer\tmale\tHer name is Anjali, she’s 5, and has a heart condition. Time is running out!\n"
            "person 1\tfemale\tPoor thing. Is there a hospital I can contact?\n"
            "scammer\tmale\tNo need, just visit our site and donate now to save her.\n"
            "person 1\tfemale\tI’d like to verify this first. Can you send documents?\n"
            "scammer\tmale\tMadam, there’s no time! Anjali won’t survive without your help.\n"
            "person 1\tfemale\tI understand, but I’ll check with the NGO directly.\n"
            "scammer\tmale\tPlease, don’t delay. Every minute counts for Anjali!\n"
            "person 1\tfemale\tI’ll call the hospital to confirm. What’s her full name?\n"
            "scammer\tmale\tAnjali Sharma. But it’s faster to pay online now!\n"
            "person 1\tfemale\tI’ll do this properly. Thanks for the info.\n"
        ],
        "Fake Delivery Scam": [
            "scammer\tfemale\tNamaste, Priya from IndiaPost. Your parcel is stuck at Delhi customs.\n"
            "person 1\tmale\tI wasn’t expecting anything. What’s this about?\n"
            "scammer\tfemale\tIt’s a gift from abroad. Pay ₹3000 at http://indpost-claim.com to release it.\n"
            "person 1\tmale\tA gift? From whom? I need details.\n"
            "scammer\tfemale\tSender is anonymous, sir. You must pay today or it’s returned.\n"
            "person 1\tmale\tThat’s odd. What’s in the parcel?\n"
            "scammer\tfemale\tElectronics, very valuable. Check our site to claim it.\n"
            "person 1\tmale\tThis doesn’t add up. Can you give me a tracking number?\n"
            "scammer\tfemale\tIt’s IPX-7890. Pay now to avoid losing it!\n"
            "person 1\tmale\tI’ll verify with IndiaPost directly. What’s your office number?\n"
            "scammer\tfemale\tSir, no time for that! Pay online or the parcel is gone.\n"
            "person 1\tmale\tI’m not paying without proof. I’ll call the post office.\n"
            "scammer\tfemale\tYou’re making a mistake, sir! It’s urgent!\n"
            "person 1\tmale\tI’ll take my chances. Goodbye.\n"
        ],
        "Financial Fraud": [
            "scammer\tmale\tSir, this is Vikram from SBI. Your account has suspicious activity.\n"
            "person 1\tfemale\tWhat? I haven’t noticed anything. What’s wrong?\n"
            "scammer\tmale\tSomeone tried to withdraw ₹50,000. Share your OTP to secure it.\n"
            "person 1\tfemale\tI didn’t get an OTP. Can you tell me the account number?\n"
            "scammer\tmale\tIt’s linked to your phone. Visit http://sbi-secure.in now.\n"
            "person 1\tfemale\tThis sounds fishy. I’ll check my account online.\n"
            "scammer\tmale\tMadam, don’t risk it! Enter your details on our site.\n"
            "person 1\tfemale\tWhy can’t you fix it from your end?\n"
            "scammer\tmale\tWe need your confirmation. It’s urgent to stop the fraud!\n"
            "person 1\tfemale\tI’m going to the bank tomorrow to sort this out.\n"
            "scammer\tmale\tThat’s too late! Your money will be gone by then!\n"
            "person 1\tfemale\tI don’t trust this call. I’m reporting it.\n"
            "scammer\tmale\tMadam, please, just follow the link!\n"
            "person 1\tfemale\tNo way. I’m hanging up.\n"
        ],
        "Identity Theft": [
            "scammer\tfemale\tNamaste, I’m from Aadhaar office. Your card is blocked due to misuse.\n"
            "person 1\tmale\tBlocked? I got no notice. What’s going on?\n"
            "scammer\tfemale\tSomeone used your Aadhaar. Update it at http://aadhaar-verify.in.\n"
            "person 1\tmale\tI’ll visit the Aadhaar center. Why call me?\n"
            "scammer\tfemale\tIt takes weeks, sir! Do it online to avoid fines.\n"
            "person 1\tmale\tFines? This sounds like a scam.\n"
            "scammer\tfemale\tNo scam, sir! Share your Aadhaar number to verify.\n"
            "person 1\tmale\tI’m not sharing anything over the phone.\n"
            "scammer\tfemale\tYou’ll face legal issues if you don’t act now!\n"
            "person 1\tmale\tSend me official documents, then we’ll talk.\n"
            "scammer\tfemale\tIt’s urgent! Just visit the link today.\n"
            "person 1\tmale\tI’m reporting this to the police.\n"
            "scammer\tfemale\tSir, you’re making a mistake!\n"
            "person 1\tmale\tGoodbye.\n"
        ],
        "Impersonation": [
            "scammer\tmale\tHello, Inspector Sharma, Delhi Police. You’re linked to a fraud case.\n"
            "person 1\tfemale\tWhat? I’ve done nothing wrong!\n"
            "scammer\tmale\tPay ₹20,000 at http://delhipolice-clear.in to clear your name.\n"
            "person 1\tfemale\tThis can’t be right. What’s the case about?\n"
            "scammer\tmale\tMoney laundering, madam. Pay now or face arrest.\n"
            "person 1\tfemale\tI need to speak to a real officer.\n"
            "scammer\tmale\tI’m real! Visit our site to avoid jail.\n"
            "person 1\tfemale\tI’m calling the police station to verify.\n"
            "scammer\tmale\tDon’t waste time! Pay today or we’ll act.\n"
            "person 1\tfemale\tI don’t trust you. What’s your badge number?\n"
            "scammer\tmale\tIt’s DP-4567. But pay now, it’s urgent!\n"
            "person 1\tfemale\tI’ll check with the station first.\n"
            "scammer\tmale\tYou’re risking everything, madam!\n"
            "person 1\tfemale\tI’m hanging up.\n"
        ],
        "Investment Scams": [
            "scammer\tfemale\tHi, Neha from WealthIndia. Invest ₹30,000, get ₹3 lakh by Diwali!\n"
            "person 1\tmale\tThat’s a huge return. How does it work?\n"
            "scammer\tfemale\tOur stock plan is foolproof. Sign up at http://wealthindia-grow.com.\n"
            "person 1\tmale\tI’ve never heard of you. Are you SEBI-registered?\n"
            "scammer\tfemale\tYes, sir! All details are on our site.\n"
            "person 1\tmale\tCan you send me a prospectus?\n"
            "scammer\tfemale\tNo need, just invest now for quick profits!\n"
            "person 1\tmale\tI don’t invest without research.\n"
            "scammer\tfemale\tThis offer ends today, sir! Don’t miss out.\n"
            "person 1\tmale\tI’ll pass. Send me documents if you’re legit.\n"
            "scammer\tfemale\tYou’re losing a golden chance!\n"
            "person 1\tmale\tI’ll take that risk. Goodbye.\n"
            "scammer\tfemale\tSir, wait, let’s discuss!\n"
            "person 1\tmale\tNo thanks.\n"
        ],
        "Job Offer Scam": [
            "scammer\tmale\tCongrats! You’re selected for Infosys. Pay ₹7000 for training at http://infosys-hire.in.\n"
            "person 1\tfemale\tI didn’t apply. How did I get selected?\n"
            "scammer\tmale\tWe found your resume online. Pay now to secure the job.\n"
            "person 1\tfemale\tThat’s strange. Can you send an offer letter?\n"
            "scammer\tmale\tIt’s on our site. Pay first to get it.\n"
            "person 1\tfemale\tNo legit company asks for payment upfront.\n"
            "scammer\tmale\tThis is standard, madam! It’s a top job.\n"
            "person 1\tfemale\tI’ll contact Infosys directly.\n"
            "scammer\tmale\tDon’t miss this chance! Pay today!\n"
            "person 1\tfemale\tI’m not paying anything. Stop calling.\n"
            "scammer\tmale\tYou’ll regret this, madam!\n"
            "person 1\tfemale\tI’m reporting this scam.\n"
            "scammer\tmale\tNo scam, just pay!\n"
            "person 1\tfemale\tGoodbye.\n"
        ],
        "Lottery Scam": [
            "scammer\tfemale\tSir, you’ve won ₹7 lakh in our Diwali lottery! Claim it at http://diwaliluck.in.\n"
            "person 1\tmale\tI didn’t enter any lottery. How’s this possible?\n"
            "scammer\tfemale\tYour number was randomly picked. Pay ₹3000 to process.\n"
            "person 1\tmale\tThat sounds like a scam. Who’s running this?\n"
            "scammer\tfemale\tIt’s a trusted company. Check our site for details.\n"
            "person 1\tmale\tI’m not paying without proof.\n"
            "scammer\tfemale\tSir, you’ll lose the prize if you delay!\n"
            "person 1\tmale\tSend me official documents, then we’ll talk.\n"
            "scammer\tfemale\tNo time for that! Pay now to claim.\n"
            "person 1\tmale\tI don’t trust this. I’m reporting it.\n"
            "scammer\tfemale\tYou’re making a mistake, sir!\n"
            "person 1\tmale\tStop calling me.\n"
            "scammer\tfemale\tLast chance, sir!\n"
            "person 1\tmale\tGoodbye.\n"
        ],
        "Loan Scam": [
            "scammer\tmale\tNamaste, I’m from QuickLoan. Get ₹2 lakh instantly at http://quickloan-india.in!\n"
            "person 1\tfemale\tI didn’t apply for a loan. Why are you calling?\n"
            "scammer\tmale\tSpecial offer for you, madam! Just share your details.\n"
            "person 1\tfemale\tI don’t need a loan. Remove my number.\n"
            "scammer\tmale\tThis is a limited offer! Apply now!\n"
            "person 1\tfemale\tI said I’m not interested.\n"
            "scammer\tmale\tMadam, you could use it for Diwali shopping!\n"
            "person 1\tfemale\tStop pushing. I’ll report this.\n"
            "scammer\tmale\tNo need to report, just a great deal!\n"
            "person 1\tfemale\tI’m blocking this number.\n"
            "scammer\tmale\tWait, let’s talk!\n"
            "person 1\tfemale\tNo. Goodbye.\n"
            "scammer\tmale\tYou’ll miss out!\n"
            "person 1\tfemale\tI said stop!\n"
        ],
        "Phishing": [
            "scammer\tfemale\tHello, your Paytm account is hacked. Share your PIN at http://paytm-secure.in.\n"
            "person 1\tmale\tI don’t use Paytm. Who is this?\n"
            "scammer\tfemale\tIt’s urgent, sir! Verify your details to stop the hack.\n"
            "person 1\tmale\tThis is a scam. I’m reporting you.\n"
            "scammer\tfemale\tNo scam, sir! Your money is at risk!\n"
            "person 1\tmale\tI don’t have a Paytm account!\n"
            "scammer\tfemale\tIt might be linked to your number. Check our site.\n"
            "person 1\tmale\tStop lying. I’m calling the police.\n"
            "scammer\tfemale\tSir, just visit the link to be safe!\n"
            "person 1\tmale\tYou’re wasting your time. Goodbye.\n"
            "scammer\tfemale\tDon’t hang up, sir!\n"
            "person 1\tmale\tI’m done here.\n"
            "scammer\tfemale\tYou’ll regret this!\n"
            "person 1\tmale\tWhatever.\n"
        ],
        "Service Fraud": [
            "scammer\tmale\tSir, your Jio connection will be cut today. Pay ₹1500 at http://jio-renew.in.\n"
            "person 1\tfemale\tI paid my bill yesterday. What’s this?\n"
            "scammer\tmale\tSystem error, madam. Pay now to stay connected.\n"
            "person 1\tfemale\tI’ll call Jio customer care.\n"
            "scammer\tmale\tNo need, just use our site. It’s faster!\n"
            "person 1\tfemale\tThis sounds like a scam.\n"
            "scammer\tmale\tNot a scam, madam! Pay to avoid disconnection.\n"
            "person 1\tfemale\tI’m checking with Jio first.\n"
            "scammer\tmale\tYou’re wasting time! Pay now!\n"
            "person 1\tfemale\tI don’t trust you. Goodbye.\n"
            "scammer\tmale\tMadam, don’t lose your connection!\n"
            "person 1\tfemale\tI said I’ll verify it.\n"
            "scammer\tmale\tLast chance!\n"
            "person 1\tfemale\tStop calling.\n"
        ],
        "Subscription Scam": [
            "scammer\tfemale\tHi, your Netflix account is expiring. Renew at http://netflix-india-pay.in for ₹600.\n"
            "person 1\tmale\tMy account is fine. I checked yesterday.\n"
            "scammer\tfemale\tThat’s an error, sir. Pay now to keep watching.\n"
            "person 1\tmale\tI’ll log in to Netflix myself.\n"
            "scammer\tfemale\tSir, it’s urgent! Renew today or lose access.\n"
            "person 1\tmale\tThis is a scam. I’m reporting it.\n"
            "scammer\tfemale\tNo scam, just a glitch! Use our site.\n"
            "person 1\tmale\tI don’t trust you. Stop calling.\n"
            "scammer\tfemale\tYou’ll miss your shows, sir!\n"
            "person 1\tmale\tI’ll handle it with Netflix directly.\n"
            "scammer\tfemale\tDon’t delay, sir!\n"
            "person 1\tmale\tGoodbye.\n"
            "scammer\tfemale\tWait, sir!\n"
            "person 1\tmale\tI’m done.\n"
        ],
        "Tech Support Scam": [
            "scammer\tmale\tHello, I’m from Microsoft. Your PC has a virus. Call 1800-123-4567 now.\n"
            "person 1\tfemale\tMy computer’s fine. How do you know this?\n"
            "scammer\tmale\tWe detected it remotely. Visit http://ms-support.in to fix it.\n"
            "person 1\tfemale\tThis sounds fake. I’m not doing that.\n"
            "scammer\tmale\tMadam, your data is at risk! Act now!\n"
            "person 1\tfemale\tI’ll run my antivirus myself.\n"
            "scammer\tmale\tThat won’t work! You need our help.\n"
            "person 1\tfemale\tI don’t trust this call.\n"
            "scammer\tmale\tYou’ll lose everything, madam!\n"
            "person 1\tfemale\tI’m reporting you. Stop calling.\n"
            "scammer\tmale\tDon’t hang up, it’s serious!\n"
            "person 1\tfemale\tGoodbye.\n"
            "scammer\tmale\tMadam, wait!\n"
            "person 1\tfemale\tI said stop.\n"
        ]
    },
    "normal": {
        "Delivery Update": [
            "person 1\tmale\tHey, my Flipkart order just got dispatched! Should arrive by tomorrow.\n"
            "person 2\tfemale\tNice! What did you get? Something for Diwali?\n"
            "person 1\tmale\tYeah, a new phone. Been waiting for this one!\n"
            "person 2\tfemale\tSweet! Which model? I’m thinking of upgrading too.\n"
            "person 1\tmale\tIt’s the latest OnePlus. Want to check it out when it arrives?\n"
            "person 2\tfemale\tTotally! Let’s unbox it over chai.\n"
            "person 1\tmale\tHaha, deal! I’ll call you when it’s here.\n"
            "person 2\tfemale\tCan’t wait. Did you get a case for it?\n"
            "person 1\tmale\tNot yet. Any recommendations?\n"
            "person 2\tfemale\tI got a good one from Amazon. I’ll send you the link.\n"
            "person 1\tmale\tAwesome, thanks! Let’s catch up tomorrow.\n"
            "person 2\tfemale\tSure thing! Have fun with the new phone.\n"
            "person 1\tmale\tWill do! Talk soon.\n"
            "person 2\tfemale\tBye!\n"
        ],
        "Social": [
            "person 1\tfemale\tHey, you free this weekend? Thinking of a movie night.\n"
            "person 2\tmale\tSounds awesome! Got any movies in mind?\n"
            "person 1\tfemale\tMaybe a Bollywood classic. Or something new?\n"
            "person 2\tmale\tLet’s go with a classic. Sholay?\n"
            "person 1\tfemale\tPerfect! My place, 7 PM Saturday?\n"
            "person 2\tmale\tWorks for me! I’ll bring popcorn.\n"
            "person 1\tfemale\tSweet! Should we invite Priya too?\n"
            "person 2\tmale\tYeah, she’d love it. I’ll text her.\n"
            "person 1\tfemale\tCool. Oh, any Diwali plans yet?\n"
            "person 2\tmale\tNot really, maybe a family dinner. You?\n"
            "person 1\tfemale\tPlanning a small party. You should come!\n"
            "person 2\tmale\tCount me in! Let’s talk details later.\n"
            "person 1\tfemale\tGreat! See you Saturday.\n"
            "person 2\tmale\tYup, bye!\n"
        ],
        "Service Inquiry": [
            "person 1\tmale\tHey, you tried that new salon in Bandra?\n"
            "person 2\tfemale\tYeah, it’s amazing! Got a haircut last week.\n"
            "person 1\tmale\tNice! I need one. How’s their service?\n"
            "person 2\tfemale\tSuper professional. Priya there is awesome.\n"
            "person 1\tmale\tGood to know. Pricey or reasonable?\n"
            "person 2\tfemale\tPretty decent, around ₹800 for a cut.\n"
            "person 1\tmale\tCool, I’ll book an appointment. Thanks!\n"
            "person 2\tfemale\tNo prob. Tell Priya I sent you.\n"
            "person 1\tmale\tWill do. You free to grab coffee after?\n"
            "person 2\tfemale\tMaybe Sunday? I’m booked tomorrow.\n"
            "person 1\tmale\tSunday works. Let’s hit that new café.\n"
            "person 2\tfemale\tSounds good! I’ll text you.\n"
            "person 1\tmale\tPerfect. Talk later!\n"
            "person 2\tfemale\tBye!\n"
        ],
        "Entertainment": [
            "person 1\tfemale\tYou watching the IPL match tonight?\n"
            "person 2\tmale\tObviously! Mumbai Indians vs. Chennai Super Kings!\n"
            "person 1\tfemale\tI’m Team Chennai. Ready to lose?\n"
            "person 2\tmale\tHaha, Mumbai’s gonna crush it! Wanna bet?\n"
            "person 1\tfemale\tSure, loser buys dinner. My place?\n"
            "person 2\tmale\tYou’re on! I’ll bring some snacks.\n"
            "person 1\tfemale\tCool. Starts at 7:30, right?\n"
            "person 2\tmale\tYup. I’m setting up my big screen.\n"
            "person 1\tfemale\tFancy! Should we call Rohan too?\n"
            "person 2\tmale\tGood idea. I’ll ping him.\n"
            "person 1\tfemale\tSweet. Got any predictions for the game?\n"
            "person 2\tmale\tMumbai by 20 runs. You?\n"
            "person 1\tfemale\tChennai by 2 wickets! We’ll see.\n"
            "person 2\tmale\tGame on! See you tonight.\n"
        ],
        "Work Update": [
            "person 1\tmale\tHey, just wrapped up that big client project!\n"
            "person 2\tfemale\tNo way! How’d it go? They happy?\n"
            "person 1\tmale\tThey loved it. Might get a bonus!\n"
            "person 2\tfemale\tThat’s awesome! Drinks to celebrate?\n"
            "person 1\tmale\tFriday night? New pub in Colaba?\n"
            "person 2\tfemale\tI’m in! Congrats again, superstar.\n"
            "person 1\tmale\tHaha, thanks. You working on anything cool?\n"
            "person 2\tfemale\tJust started a new campaign. It’s hectic.\n"
            "person 1\tmale\tYou’ll nail it. Need a coffee break?\n"
            "person 2\tfemale\tTempting! Maybe tomorrow afternoon?\n"
            "person 1\tmale\tWorks for me. Usual spot?\n"
            "person 2\tfemale\tYup. I’ll text you.\n"
            "person 1\tmale\tCool. Catch you later!\n"
            "person 2\tfemale\tBye!\n"
        ],
        "Family": [
            "person 1\tfemale\tMa, I’m coming home for Diwali next week!\n"
            "person 2\tmale\tThat’s great, beta! We’re so excited.\n"
            "person 1\tfemale\tCan’t wait for your laddoos. How’s Papa?\n"
            "person 2\tmale\tHe’s good, planning the puja already.\n"
            "person 1\tfemale\tHaha, classic Papa. Anyone else coming?\n"
            "person 2\tmale\tYour brother might visit too.\n"
            "person 1\tfemale\tNice! I’ll call him to confirm.\n"
            "person 2\tmale\tGood idea. What time’s your train?\n"
            "person 1\tfemale\tReaches at 8 AM Tuesday. Pick me up?\n"
            "person 2\tmale\tOf course, beta. We’ll be there.\n"
            "person 1\tfemale\tThanks, Ma! I’m bringing some gifts.\n"
            "person 2\tmale\tYou don’t have to, just come home!\n"
            "person 1\tfemale\tHaha, see you soon!\n"
            "person 2\tmale\tTake care, beta.\n"
        ],
        "Sports": [
            "person 1\tmale\tHey, you up for cricket this Sunday?\n"
            "person 2\tfemale\tTotally! Same ground, 7 AM?\n"
            "person 1\tmale\tYup. I’m bringing my new bat.\n"
            "person 2\tfemale\tFancy! Ready to lose, though?\n"
            "person 1\tmale\tHaha, I’m in top form. You’re going down!\n"
            "person 2\tfemale\tWe’ll see! Who else is playing?\n"
            "person 1\tmale\tRohan and a few others. Good crew.\n"
            "person 2\tfemale\tSweet. Drinks after the game?\n"
            "person 1\tmale\tDefinitely. Loser buys, deal?\n"
            "person 2\tfemale\tDeal! I’m picking the place.\n"
            "person 1\tmale\tFair enough. Practice your bowling!\n"
            "person 2\tfemale\tOh, it’s on! See you Sunday.\n"
            "person 1\tmale\tBring it! Bye.\n"
            "person 2\tfemale\tLater!\n"
        ],
        "Recreation": [
            "person 1\tfemale\tI’m planning a trek to Lonavala next weekend. You in?\n"
            "person 2\tmale\tThat sounds epic! When exactly?\n"
            "person 1\tfemale\tSaturday morning. Train leaves at 6 AM.\n"
            "person 2\tmale\tEarly, but I’m game. Who else is coming?\n"
            "person 1\tfemale\tJust us and maybe Priya. Small group.\n"
            "person 2\tmale\tNice. I’ll pack my hiking boots.\n"
            "person 1\tfemale\tCool. I’m booking tickets tomorrow.\n"
            "person 2\tmale\tLet me know the cost. Need any gear?\n"
            "person 1\tfemale\tJust a backpack and water. I’ll send a list.\n"
            "person 2\tmale\tPerfect. Been a while since we trekked!\n"
            "person 1\tfemale\tRight? It’s gonna be fun.\n"
            "person 2\tmale\tTotally. Let’s plan dinner after.\n"
            "person 1\tfemale\tGood call. I’ll pick a spot.\n"
            "person 2\tmale\tSee you soon!\n"
        ],
        "Education": [
            "person 1\tmale\tHey, I just passed my IIT entrance exam!\n"
            "person 2\tfemale\tThat’s huge! Congrats, genius!\n"
            "person 1\tmale\tThanks! Still can’t believe it.\n"
            "person 2\tfemale\tYou worked so hard. What’s next?\n"
            "person 1\tmale\tCounseling rounds, then choosing a branch.\n"
            "person 2\tfemale\tExciting! Got a preference?\n"
            "person 1\tmale\tLeaning toward Computer Science.\n"
            "person 2\tfemale\tSolid choice. Wanna celebrate this weekend?\n"
            "person 1\tmale\tYeah, dinner sounds great. Your treat?\n"
            "person 2\tfemale\tHaha, fine, you win this time!\n"
            "person 1\tmale\tSweet! Pick a place.\n"
            "person 2\tfemale\tHow about that new restaurant in Juhu?\n"
            "person 1\tmale\tPerfect. I’ll book a table.\n"
            "person 2\tfemale\tAwesome. So proud of you!\n"
        ],
        "Travel": [
            "person 1\tfemale\tJust booked tickets to Goa for New Year!\n"
            "person 2\tmale\tNo way! That’s gonna be epic. When?\n"
            "person 1\tfemale\tDec 30th to Jan 3rd. Beach vibes!\n"
            "person 2\tmale\tJealous! You going solo or with friends?\n"
            "person 1\tfemale\tWith Rohan and Priya. Wanna join?\n"
            "person 2\tmale\tTempting! How much are flights?\n"
            "person 1\tfemale\tGot them for ₹15,000 round trip. Not bad.\n"
            "person 2\tmale\tNice deal. Let me check my schedule.\n"
            "person 1\tfemale\tCool, let me know soon. We’re booking a villa.\n"
            "person 2\tmale\tFancy! What’s the plan there?\n"
            "person 1\tfemale\tBeaches, parties, maybe some water sports.\n"
            "person 2\tmale\tSounds perfect. I’ll confirm by tomorrow.\n"
            "person 1\tfemale\tGreat! It’ll be a blast.\n"
            "person 2\tmale\tBet it will. Talk soon!\n"
        ]
    }
}


# Function to create a batch prompt for call transcripts
def create_batch_prompt(type, category, num_examples=2, batch_size=3):
    selected = random.sample(
        EXAMPLE_PROMPTS[type][category],
        min(num_examples, len(EXAMPLE_PROMPTS[type][category])))
    genders = ["male", "female"]
    if type == "fraud":
        prompt = (
            f"Generate {batch_size} realistic fraud call transcripts for {category} in the Indian context. "
            f"Each transcript must have 10-20 turns (5-10 per speaker), with scammer (gender: {random.choice(genders)}) and person 1 (gender: {random.choice(genders)}). "
            f"Include urgent, suspicious elements like fake URLs (e.g., http://secure.sbi-verify.com), phone numbers, or emotional triggers. "
            f"Use Indian-specific references (e.g., SBI, Aadhaar, Diwali). "
            f"Ensure transcripts are unique, varied, and conversational, with natural greetings, pauses, and realistic dialogue. "
            f"Each line must be formatted as 'scammer\tgender\tmessage' or 'person 1\tgender\tmessage'. "
            f"Transcripts must alternate speakers, starting with 'scammer'. "
            f"Return a numbered list of transcripts, e.g., '1:', followed by tab-separated lines. "
            f"Base them on these examples:\n")
    else:
        prompt = (
            f"Generate {batch_size} realistic normal call transcripts for {category} in the Indian context. "
            f"Each transcript must have 10-20 turns (5-10 per speaker), with person 1 (gender: {random.choice(genders)}) and person 2 (gender: {random.choice(genders)}). "
            f"Use casual, friendly language typical of phone calls, like updates or social chats. "
            f"Include Indian-specific references (e.g., Diwali, Mumbai, chai). "
            f"Ensure transcripts are unique, varied, and conversational, with natural greetings, pauses, and realistic dialogue. "
            f"Each line must be formatted as 'person 1\tgender\tmessage' or 'person 2\tgender\tmessage'. "
            f"Transcripts must alternate speakers, starting with 'person 1'. "
            f"Return a numbered list of transcripts, e.g., '1:', followed by tab-separated lines. "
            f"Base them on these examples:\n")

    for transcript in selected:
        prompt += "\nTranscript:\n"
        prompt += transcript
        prompt += "\n"
    prompt += "New call transcripts (numbered list):"
    return prompt


# Function to parse batch response
def parse_batch_response(response, batch_size):
    if isinstance(response, dict):
        text = response.get("content", "").strip()
    else:
        text = response.content.strip()

    transcripts = []
    current_transcript = []
    current_number = None
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if re.match(r"^\d+[.:]\s*$", line):
            if current_transcript:
                if len(current_transcript) >= 10:  # Minimum 10 turns
                    transcripts.append(current_transcript)
                else:
                    print(
                        f"Discarded transcript {current_number}: only {len(current_transcript)} turns"
                    )
            current_transcript = []
            current_number = int(re.match(r"^\d+", line).group())
            i += 1
        elif line and current_number:
            parts = line.split("\t")
            if len(parts) == 3:
                speaker, gender, message = parts
                if gender not in ["male", "female"]:
                    print(f"Invalid gender in line: {line}")
                    i += 1
                    continue
                # Ensure correct speaker sequence
                if ((not current_transcript and
                     speaker in ["scammer", "person 1"]) or
                    (current_transcript and
                     current_transcript[0][0] == "scammer" and
                     speaker in ["scammer", "person 1"]) or
                    (current_transcript and
                     current_transcript[0][0] == "person 1" and
                     speaker in ["person 1", "person 2"])):
                    # Check for speaker alternation
                    if not current_transcript or (current_transcript and
                                                  current_transcript[-1][0]
                                                  != speaker):
                        current_transcript.append((speaker, gender, message))
                    else:
                        print(f"Non-alternating speaker in line: {line}")
                else:
                    print(f"Invalid speaker in line: {line}")
            else:
                print(f"Malformed line: {line}")
            i += 1
        else:
            i += 1
    if current_transcript and len(current_transcript) >= 10:
        transcripts.append(current_transcript)
    elif current_transcript:
        print(
            f"Discarded final transcript: only {len(current_transcript)} turns")

    token_usage = response.response_metadata.get(
        "token_usage", {}) if isinstance(response, dict) else {}
    if len(transcripts) < batch_size:
        print(
            f"Generated {len(transcripts)} of {batch_size} transcripts. Token usage: {token_usage}"
        )
    return transcripts, token_usage


# Function to generate transcripts with backoff
@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def generate_batch_transcripts(api_key, prompt, batch_size):
    llm = ChatOpenAI(api_key=api_key,
                     base_url="https://api.x.ai/v1",
                     model="grok-3",
                     temperature=0.9,
                     max_tokens=3000)
    prompt_template = ChatPromptTemplate.from_messages([("human", "{prompt}")])
    llm_chain = prompt_template | llm
    response = llm_chain.invoke({"prompt": prompt})
    return parse_batch_response(response, batch_size)


# Helper function to generate unique transcripts for a category
def generate_category_transcripts(type,
                                  category,
                                  num_required,
                                  global_generated_set,
                                  api_key,
                                  batch_size=3):
    generated_transcripts = []
    max_attempts = 5
    attempts = 0
    total_tokens = 0
    while len(generated_transcripts) < num_required and attempts < max_attempts:
        prompt = create_batch_prompt(type,
                                     category,
                                     num_examples=2,
                                     batch_size=batch_size)
        try:
            transcripts, token_usage = generate_batch_transcripts(
                api_key, prompt, batch_size)
            for transcript in transcripts:
                transcript_str = "\n".join(
                    [f"{s}\t{g}\t{m}" for s, g, m in transcript])
                if (len(generated_transcripts) < num_required and
                        transcript_str not in global_generated_set and
                        len(transcript) >= 10):
                    generated_transcripts.append(transcript)
                    global_generated_set.add(transcript_str)
            attempts += 1
            total_tokens += token_usage.get("total_tokens", 0)
            if len(transcripts) == 0:
                print(
                    f"No valid transcripts generated for {type}/{category} on attempt {attempts}"
                )
        except Exception as e:
            print(f"Error generating transcripts for {type}/{category}: {e}")
            time.sleep(1)
            attempts += 1
    if len(generated_transcripts) < num_required:
        print(
            f"Only generated {len(generated_transcripts)} of {num_required} transcripts for {type}/{category}. Total tokens: {total_tokens}"
        )
    return [{
        "type":
            type,
        "category":
            category,
        "transcript": [{
            "speaker": s,
            "gender": g,
            "message": m
        } for s, g, m in trans]
    } for trans in generated_transcripts]


# Function to generate a single dataset
def generate_dataset(api_key, num_entries_per_category, global_generated_set):
    dataset = []

    scam_categories = [
        "Emotional Manipulation", "Fake Delivery Scam", "Financial Fraud",
        "Identity Theft", "Impersonation", "Investment Scams", "Job Offer Scam",
        "Lottery Scam", "Loan Scam", "Phishing", "Service Fraud",
        "Subscription Scam", "Tech Support Scam"
    ]
    normal_categories = [
        "Delivery Update", "Social", "Service Inquiry", "Entertainment",
        "Work Update", "Family", "Sports", "Recreation", "Education", "Travel"
    ]

    categories = [
        (type, cat)
        for type in ["fraud", "normal"]
        for cat in (scam_categories if type == "fraud" else normal_categories)
    ]
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(generate_category_transcripts, type, cat,
                            num_entries_per_category, global_generated_set,
                            api_key) for type, cat in categories
        ]
        for future in futures:
            try:
                dataset.extend(future.result())
            except Exception as e:
                print(f"Error in future result: {e}")

    random.shuffle(dataset)
    return dataset


if __name__ == "__main__":
    # load the API key from the config file
    api_key = misc.get_api_key(f"{CONFIG_DIR}/{API_KEY_FILE_NAME}",
                               "GROK_API_KEY")
    global_generated_set = set()

    num_datasets = 5
    num_entries_per_category = 5
    for i in range(num_datasets):
        start_time = time.time()
        dataset = generate_dataset(api_key, num_entries_per_category,
                                   global_generated_set)
        output_file = f"{DATASET_OUTPUT_DIR}/{DATASET_OUTPUT_BASE_NAME}_{i}.json"
        misc.save_json_file(output_file, dataset)
