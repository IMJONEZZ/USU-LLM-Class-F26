"""Build per-question targeted training examples for Assignment 6.

Each group of examples practises the *kind* of reasoning one assignment question
needs, using different scenarios, objects, numbers and wording. Exact assignment prompts are excluded; semantic similarity
and instructor permission still require review. Arithmetic, spelling and
counting answers are computed by code; the rest are short hand-written answers that
the student reviews.

Run from the repository root: ``uv run python assignment_6/make_targeted_dataset.py``.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

OUTPUT = Path(__file__).with_name("targeted_examples.jsonl")


def qa(tag: str, question: str, answer: str) -> dict:
    return {
        "tag": tag,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }


# ---------------------------------------------------------------- Q1
# Fixed-duration events: more people do not shorten them.
Q1_FIXED = [
    (
        "A quartet performs a 6-minute song. If the band grows to 40 members, how long does the song take?",
        "6 minutes. The song lasts as long as the music is written to last, so adding musicians does not make it faster.",
    ),
    (
        "A choir of 20 sings a 5-minute anthem. How long will a choir of 200 take to sing it?",
        "5 minutes. The anthem has a fixed length, so a bigger choir sings it in the same time.",
    ),
    (
        "10 actors perform a 90-minute play. How long does the show run if the cast is 100 actors?",
        "90 minutes. The play's running time is set by the script and the pacing, not by the size of the cast.",
    ),
    (
        "If 30 students sit through a 50-minute lecture, how long is the lecture for 300 students?",
        "50 minutes. The lecture takes the same time whether 30 or 300 people listen.",
    ),
    (
        "A 2-hour film is watched by 50 viewers. How long does it take 500 viewers to watch it?",
        "2 hours. Everyone watches at the same time, so more viewers do not shorten the film.",
    ),
    (
        "8 dancers perform a 4-minute routine. How long will it take 80 dancers to perform it?",
        "4 minutes. The routine follows the music, so a larger group takes just as long.",
    ),
    (
        "A train ride takes 3 hours with 5 passengers on board. How long will it take with 500 passengers?",
        "3 hours. The journey time depends on the route and the train's speed, not the number of passengers.",
    ),
    (
        "If 15 commuters take a 40-minute bus ride, how long does the ride take for 150 commuters?",
        "40 minutes. The bus covers the same route at the same speed, whoever is on it.",
    ),
    (
        "5 people hold a 20-minute meditation session. How long would the session be for 50 people?",
        "20 minutes. The session's length is set in advance, so more participants do not change it.",
    ),
    (
        "A video call lasts 5 minutes with 4 participants. How long will it last with 40 participants?",
        "5 minutes. The length of the call depends on the conversation, not the head count.",
    ),
    (
        "It takes 10 minutes to boil 2 eggs in a large pot. How long will it take to boil 12 eggs in the same pot?",
        "About 10 minutes. The eggs cook at the same time in the same hot water, so the time is the same as long as they all fit and the water stays boiling.",
    ),
    (
        "An oven bakes 4 loaves in 30 minutes. If a bigger oven holds 16 loaves, how long will it take to bake them all?",
        "About 30 minutes. The loaves bake at the same time, so the time stays the same as long as the oven is big enough.",
    ),
    (
        "A string quartet of 4 takes 25 minutes to play a piece. How long does a full orchestra of 80 take to play the same piece?",
        "25 minutes. The piece has a fixed tempo and length, so the size of the group does not change the duration.",
    ),
    (
        "It takes a letter 3 days to travel by post to another city. How long will 300 letters sent together take?",
        "About 3 days. The letters travel through the same system at the same time, so sending more does not make delivery faster or slower.",
    ),
    (
        "If 12 people attend a 45-minute yoga class, how long is the class when 120 people attend?",
        "45 minutes. The class is scheduled for a set time regardless of how many people join.",
    ),
    (
        "A flight from one city to another takes 4 hours with 100 passengers. How long would it take with 300 passengers?",
        "About 4 hours. The flight time depends on the distance and the aircraft's speed, not the number of passengers.",
    ),
    (
        "A play-by-play radio commentary of a 90-minute match is heard by 1,000 listeners. How long will it last if 100,000 listeners tune in?",
        "90 minutes. The match and commentary last the same time however many people listen.",
    ),
    (
        "2 friends watch the sun set in 25 minutes. How long does the sunset take if 200 people watch it?",
        "25 minutes. The sunset takes the same time no matter how many people watch.",
    ),
    (
        "A 10-minute fireworks show is seen by 100 people. How long does it last for 10,000 people?",
        "10 minutes. The show is timed in advance, so the crowd size does not change it.",
    ),
    (
        "A cake needs 40 minutes in the oven. If you bake 3 cakes together on separate shelves, how long does it take?",
        "About 40 minutes. They bake at the same time, so baking more at once does not add time as long as the oven can hold them.",
    ),
    (
        "A kettle boils a litre of water in 4 minutes. How long does it take to boil 3 litres in a kettle big enough for it?",
        "More than 4 minutes. Unlike the earlier cases, heating more water takes more energy, so it takes roughly three times as long with the same heater.",
    ),
]

# Work that shortens when more workers share it (so the model learns the difference).
Q1_PROPORTIONAL = [
    ("painters", "paint a fence", "hours", 4, 6, 8),
    ("diggers", "dig a trench", "hours", 3, 12, 6),
    ("bakers", "decorate 24 cakes", "hours", 2, 8, 4),
    ("volunteers", "stuff 600 envelopes", "hours", 6, 10, 15),
    ("movers", "load a truck", "hours", 5, 8, 10),
    ("farmers", "harvest a field", "days", 10, 6, 20),
    ("typists", "type up a long report", "hours", 4, 9, 12),
    ("gardeners", "weed a large garden", "hours", 3, 10, 5),
    ("students", "fold 300 paper cranes", "hours", 8, 6, 12),
    ("cleaners", "clean an office building", "hours", 2, 9, 6),
    ("bricklayers", "build a garden wall", "days", 6, 8, 12),
    ("proofreaders", "check a long manuscript", "days", 2, 12, 8),
    ("volunteers", "sort a pile of donated clothes", "hours", 5, 6, 10),
    ("workers", "pack 1,000 parcels", "hours", 10, 9, 15),
]
Q1_TEMPLATES = [
    "If {n1} {who} can {task} in {t1} {unit}, how long will it take {n2} {who} working at the same pace?",
    "{n1} {who} take {t1} {unit} to {task}. How long would {n2} {who} need, all working at the same pace?",
]


def q1_examples() -> list[dict]:
    rows = [qa("Q1", q, a) for q, a in Q1_FIXED]
    for i, (who, task, unit, n1, t1, n2) in enumerate(Q1_PROPORTIONAL):
        t2 = n1 * t1 / n2
        assert t2 == int(t2)
        t2 = int(t2)
        work = n1 * t1
        question = Q1_TEMPLATES[i % 2].format(
            n1=n1, who=who, task=task, t1=t1, unit=unit, n2=n2
        )
        answer = (
            f"{t2} {unit}. This job can be shared, so the total work is {n1} x {t1} = "
            f"{work} worker-{unit}, and {n2} {who} finish it in {work} / {n2} = {t2} {unit}."
        )
        rows.append(qa("Q1", question, answer))
    rows += [
        qa(
            "Q1",
            "If 5 machines take 5 minutes to make 5 gadgets, how long would 100 machines take to make 100 gadgets?",
            "5 minutes. Each machine makes one gadget in 5 minutes, so 100 machines make 100 gadgets in the same 5 minutes.",
        ),
        qa(
            "Q1",
            "It takes 3 cooks 3 hours to make 3 pots of stew, one pot each. How long would 9 cooks take to make 9 pots?",
            "3 hours. Each cook makes one pot in 3 hours, so 9 cooks make 9 pots in the same time.",
        ),
        qa(
            "Q1",
            "A tap fills a bucket in 2 minutes. How long do 10 identical taps filling 10 separate buckets take?",
            "2 minutes. Each tap fills its own bucket at the same time, so the time stays 2 minutes.",
        ),
        qa(
            "Q1",
            "A tap fills one tank in 2 hours. If 4 identical taps fill that same tank together, how long does it take?",
            "30 minutes. Together they fill four times as fast, so 2 hours divided by 4 is 30 minutes.",
        ),
        qa(
            "Q1",
            "One printer prints 120 pages in 30 minutes. How long do 3 identical printers take to print 120 pages between them?",
            "10 minutes. Splitting the pages across 3 printers makes the job 3 times faster, so 30 minutes divided by 3 is 10 minutes.",
        ),
        qa(
            "Q1",
            "A book club of 6 reads a 300-page novel in 10 days, each member reading the whole book. How long will it take if 60 people join, each reading the whole book?",
            "10 days. Each person still has to read the whole novel at their own pace, so more readers do not make it faster.",
        ),
    ]
    return rows


# ---------------------------------------------------------------- Q2
Q2_EQUAL_PAIRS = [
    ("cotton", "iron"), ("bubble wrap", "bricks"), ("sand", "water"), ("popcorn", "marbles"),
    ("flour", "cement"), ("hay", "steel"), ("paper", "coal"), ("balloons", "bowling balls"),
    ("sponges", "granite"), ("dandelion fluff", "gravel"),
]  # fmt: skip
Q2_UNITS = ["kilogram", "gram", "metric ton"]

Q2_AMBIGUOUS = [
    (
        "Which is heavier, a stone of potatoes or a stone from a garden wall?",
        "It depends on what 'stone' means. As a unit of weight, a stone is about 6.35 kg, so a stone of potatoes weighs that much. A stone from a wall is just a rock and could weigh far more or less, so the two can't be compared unless you say which meaning you intend.",
    ),
    (
        "Is a metric ton heavier than a US ton?",
        "Yes. A metric ton is 1,000 kg, while a US (short) ton is about 907 kg.",
    ),
    (
        "Which is heavier, a metric ton or a long ton?",
        "The long ton. A long ton, used in the UK, is about 1,016 kg, a little more than a metric ton of 1,000 kg.",
    ),
    (
        "Is an ounce of water or a fluid ounce of water heavier?",
        "A fluid ounce of water is slightly heavier. A fluid ounce measures volume (about 29.6 mL, which weighs about 29.6 g), while an ounce of weight is about 28.35 g.",
    ),
    (
        "Is a carat of diamond heavier than a karat of gold?",
        "They can't be compared that way. A carat is a unit of mass (200 mg) used for gems, while a karat measures the purity of gold (24-karat is pure) and is not a weight.",
    ),
    (
        "Which weighs more, a US penny or a pennyweight?",
        "A US penny. A modern US penny weighs about 2.5 g, while a pennyweight is a troy unit of about 1.56 g.",
    ),
    (
        "Is a troy ounce heavier than an ordinary ounce?",
        "Yes. A troy ounce is about 31.1 g, while an ordinary (avoirdupois) ounce is about 28.35 g.",
    ),
    (
        "Which is heavier, a dollar or a kilogram of rice?",
        "A dollar is money, not a weight, so the question needs a clearer meaning. If you mean a one-dollar bill, it weighs about 1 g, so a kilogram of rice is far heavier.",
    ),
    (
        "Is a gram of gold heavier than a gram of silver?",
        "No. A gram is a gram, so they weigh the same. Gold is denser, so it takes up less space.",
    ),
    (
        "Which weighs more, a kilo of sugar or a kilogram of salt?",
        "They weigh the same. A kilo is short for kilogram, so both are 1 kilogram.",
    ),
    (
        "Is a ton of coal heavier than 2,000 kg of cotton?",
        "No. Whichever ton you mean (a metric ton is 1,000 kg, a US ton is about 907 kg and a long ton is about 1,016 kg), it is well under 2,000 kg, so the 2,000 kg of cotton is heavier.",
    ),
    (
        "Which is heavier, a mile of string or a kilometer of the same string?",
        "The mile of string. A mile is about 1.61 km, so it is longer than a kilometer and the same string weighs more.",
    ),
    (
        "Is a pint of water or a kilogram of water heavier?",
        "The kilogram. A pint of water weighs only about 0.47 to 0.57 kg, depending on whether it is a US or UK pint.",
    ),
    (
        "Is a liter of water heavier than a kilogram of water?",
        "No. They are essentially the same: a liter of water weighs about one kilogram.",
    ),
]

# (coin, mass in grams)
COINS = [
    ("one-euro coin", 7.5), ("two-euro coin", 8.5), ("US quarter", 5.67), ("US nickel", 5.0),
    ("one-yen coin", 1.0), ("100-yen coin", 4.8), ("500-yen coin", 7.1), ("Swiss five-franc coin", 13.2),
    ("Canadian one-dollar coin", 6.27), ("Australian one-dollar coin", 9.0), ("20-cent euro coin", 5.74),
    ("US dime", 2.27),
]  # fmt: skip
COIN_MONEY = {
    "one-euro coin": "1 euro", "two-euro coin": "2 euros", "US quarter": "25 cents",
    "US nickel": "5 cents", "one-yen coin": "1 yen", "100-yen coin": "100 yen",
    "500-yen coin": "500 yen", "Swiss five-franc coin": "5 francs",
    "Canadian one-dollar coin": "a Canadian dollar", "Australian one-dollar coin": "an Australian dollar",
    "20-cent euro coin": "20 euro cents", "US dime": "10 cents",
}  # fmt: skip
MASSES = [
    ("a 500 g bag of flour", 500.0),
    ("an apple of about 150 g", 150.0),
    ("a 1 kg bag of sugar", 1000.0),
]


def q2_examples() -> list[dict]:
    rows = []
    for i, (bulky, dense) in enumerate(Q2_EQUAL_PAIRS):
        unit = Q2_UNITS[i % 3]
        a = bulky if i % 2 == 0 else dense
        b = dense if i % 2 == 0 else bulky
        rows.append(
            qa(
                "Q2",
                f"Which is heavier, a {unit} of {a} or a {unit} of {b}?",
                f"Neither. They both weigh one {unit}. The {bulky} just takes up much more space than the {dense}.",
            )
        )
    rows += [qa("Q2", q, a) for q, a in Q2_AMBIGUOUS]
    for i, (coin, grams) in enumerate(COINS):
        name, mass = MASSES[i % 3]
        money = COIN_MONEY[coin]
        heavier = name if mass > grams else f"the {coin}"
        question = (
            f"Is {money} heavier than {name}?"
            if i % 2 == 0
            else f"Which weighs more, {money} or {name}?"
        )
        answer = (
            f"An amount of money isn't a weight, so the question needs a clearer meaning. "
            f"If you mean the physical {coin}, it weighs about {grams:g} g, so {heavier} is heavier."
        )
        rows.append(qa("Q2", question, answer))
    rows += [
        qa(
            "Q2",
            "Is a kilogram of steel heavier than 1,000 grams of steel?",
            "No. 1,000 grams is exactly 1 kilogram, so they weigh the same.",
        ),
        qa(
            "Q2",
            "Which is heavier, 2 kilograms of lettuce or 2,000 grams of bricks?",
            "Neither. 2 kilograms is the same as 2,000 grams, so they weigh the same. The lettuce just takes up more room.",
        ),
        qa(
            "Q2",
            "Is a ten-dollar bill heavier than ten kilograms of apples?",
            "A ten-dollar amount is money, not a weight. A ten-dollar bill itself weighs only about 1 g, so ten kilograms of apples is far heavier.",
        ),
        qa(
            "Q2",
            "Which is heavier, a gram of lead or a gram of helium?",
            "They weigh the same. A gram is a gram. The helium would take up far more space, because it is much less dense.",
        ),
    ]
    return rows


# ---------------------------------------------------------------- Q3
Q3 = [
    (
        "A girl wakes up to the smell of pancakes and finds balloons, streamers and a cake with candles in the kitchen. What's going on?",
        "It's her birthday. The cake with candles, balloons and streamers are classic birthday decorations.",
    ),
    (
        "A man comes home and finds his kitchen full of relatives, a huge roasted turkey, and pumpkin pie on the counter. What is going on?",
        "It's probably Thanksgiving, which is celebrated in the US in late November. The family gathering, the roasted turkey and the pumpkin pie are traditional parts of it.",
    ),
    (
        "Children in costumes are knocking on doors and getting candy from a neighbor with carved pumpkins on the porch. What's happening?",
        "It's Halloween. Costumes, trick-or-treating and carved pumpkins are its signature traditions.",
    ),
    (
        "A little boy hunts through the garden for painted eggs and finds a basket with chocolate bunnies. What's going on?",
        "It's Easter morning. Hidden painted eggs, baskets and chocolate bunnies are Easter traditions.",
    ),
    (
        "At midnight, everyone at a party counts down from ten, throws confetti and cheers. What's going on?",
        "It's New Year's Eve. The countdown to midnight and the confetti mark the start of the new year.",
    ),
    (
        "A woman receives a bouquet of red roses and a heart-shaped card from her husband. What's going on?",
        "It's probably Valentine's Day, February 14, when people give roses and heart cards to those they love.",
    ),
    (
        "Guests in formal clothes sit in rows while a person in a white dress walks down an aisle to music. What's going on?",
        "It's a wedding. The white dress, the aisle and the guests in formal clothes are typical of a wedding ceremony.",
    ),
    (
        "Students in caps and gowns walk across a stage, one by one, and receive rolled papers from a person at a podium. What's going on?",
        "It's a graduation ceremony. The caps and gowns and the handing out of diplomas are what happens at graduation.",
    ),
    (
        "A family lights a row of small oil lamps outside their home, draws colorful patterns on the floor and shares sweets. What's going on?",
        "It's Diwali, the Hindu festival of lights. Oil lamps (diyas), colorful floor patterns called rangoli and sweets are traditional.",
    ),
    (
        "A family lights a candle each night on a special candelabrum with nine branches and plays a game with a spinning top. What's going on?",
        "It's Hanukkah, the Jewish festival of lights. The menorah has nine branches, and playing dreidel, a spinning top game, is a traditional part of it.",
    ),
    (
        "After a month of fasting during daylight, a family puts on new clothes, prays together and shares a big feast. What's going on?",
        "It's Eid al-Fitr, the celebration marking the end of Ramadan, the month of fasting.",
    ),
    (
        "Children get small red envelopes with money, there are lion dances in the street and families eat dumplings together. What's going on?",
        "It's Lunar New Year (Chinese New Year). Red envelopes, lion dances and dumplings are traditional.",
    ),
    (
        "Crowds gather at dusk to watch fireworks, wave flags and eat barbecue on a hot July day in the US. What's going on?",
        "It's Independence Day, the Fourth of July, which celebrates the US declaration of independence in 1776.",
    ),
    (
        "A child puts on a new backpack, takes a lunchbox and walks to a building where other kids are lining up. What's going on?",
        "It's probably the first day of school. The new backpack, the lunchbox and the other children lining up are typical of it.",
    ),
    (
        "Two children carry a tray with breakfast and a hand-drawn card into their mother's bedroom. What's going on?",
        "It's probably Mother's Day. Breakfast in bed and a handmade card are a classic way to celebrate it.",
    ),
    (
        "People in green clothes, with shamrock decorations, watch a parade in the street in March. What's going on?",
        "It's St. Patrick's Day, March 17. Green clothes, shamrocks and parades are the traditions of the day.",
    ),
    (
        "A city street fills with floats, people in masks and colorful beads, and there's a ring-shaped cake with a small plastic baby inside. What's going on?",
        "It's Mardi Gras. The parade floats, masks, beads and king cake with a hidden plastic baby are its traditions.",
    ),
    (
        "Families gather under a full moon, eat round pastries filled with sweet paste and light lanterns. What's going on?",
        "It's the Mid-Autumn Festival. Mooncakes, lanterns and moon-gazing are its traditions.",
    ),
    (
        "A family sets a table with a special plate holding a roasted bone, bitter herbs and flat unleavened bread, and retells an old story. What's going on?",
        "It's a Passover seder. The seder plate, bitter herbs and matzah (unleavened bread) are part of the meal that retells the Exodus story.",
    ),
    (
        "A family decorates a small table with orange marigolds, photos of loved ones who have died, and sugar skulls. What's going on?",
        "It's Day of the Dead (Día de los Muertos) in Mexico. Marigolds, photos and sugar skulls on an altar called an ofrenda honor deceased loved ones.",
    ),
    (
        "Dozens of friends hide in the dark living room, and when the birthday person walks in they all jump out and shout. What's going on?",
        "It's a surprise party. The hiding and the shout when the person walks in are the hallmark of one.",
    ),
    (
        "The lights go out in a whole neighborhood, and neighbors are lighting candles and checking flashlights. What's going on?",
        "It's a power outage. The lights going out everywhere and people using candles and flashlights point to a blackout.",
    ),
    (
        "A kid wakes up, looks outside at deep snow, and hears on the radio that all the schools are closed. What's going on?",
        "It's a snow day. The deep snow and the announcement that schools are closed mean classes are canceled.",
    ),
    (
        "Tables on a front lawn hold old toys, books and kitchen items with price stickers, and people are browsing and bargaining. What's going on?",
        "It's a garage sale (yard sale), where people sell used household items cheaply.",
    ),
    (
        "Thousands of people in running shoes with numbers pinned to their shirts pour through closed-off city streets while crowds cheer. What's going on?",
        "It's a marathon or a big road race. The numbered bibs, the closed streets and the cheering crowds are signs of one.",
    ),
    (
        "A family pitches a tent, builds a fire and roasts marshmallows under the stars. What's going on?",
        "They're camping. The tent, the campfire and the roasted marshmallows are classic parts of a camping trip.",
    ),
    (
        "A big crowd in team jerseys eats snacks around a TV, cheering and groaning at a football game. What's going on?",
        "It's probably a Super Bowl party or a similar big game watch party. The jerseys, snacks and shared TV are typical.",
    ),
    (
        "Long tables sit under a huge tent with people in traditional outfits raising big mugs of beer in Munich. What's going on?",
        "It's Oktoberfest, the big annual beer festival in Munich.",
    ),
    (
        "In France, crowds line the streets on July 14 to watch a military parade and fireworks. What's going on?",
        "It's Bastille Day, France's national day, which marks the storming of the Bastille in 1789.",
    ),
    (
        "A classroom is full of tri-fold poster boards with volcano models, plant experiments and kids explaining their projects to judges. What's going on?",
        "It's a school science fair, where students present projects and judges evaluate them.",
    ),
    (
        "Friends and family gather around a table with a cake, give a retiring coworker a card and a gift, and make speeches about their career. What's going on?",
        "It's a retirement party for the coworker who is leaving the workforce.",
    ),
    (
        "A couple dresses up, goes to a candlelit restaurant and the waiter brings a small cake with a single candle. What's going on?",
        "They're probably celebrating an anniversary or a special occasion, such as a birthday. The candlelit dinner and the small cake with a candle suggest it.",
    ),
    (
        "Guests bring pastel gifts for a woman with a round belly, and they play games like guessing baby foods. What's going on?",
        "It's a baby shower, where friends and family give gifts and celebrate an upcoming birth.",
    ),
    (
        "A small town's main street has tables of homemade cookies and cupcakes with a sign about raising money for a school trip. What's going on?",
        "It's a bake sale, a fundraiser where baked goods are sold to raise money.",
    ),
    (
        "A woman in a tall chef's hat is surrounded by cameras and a panel of judges tasting dishes while the clock counts down. What's going on?",
        "It looks like a cooking competition, such as a televised cooking show. The cameras, judges, timer and chef's hat are the clues.",
    ),
    (
        "An entire neighborhood is out on the sidewalks in the afternoon, kids on bikes, music playing, and tables of food are set up in the street. What's going on?",
        "It's probably a block party, where neighbors close a street to socialize, eat and play together.",
    ),
    (
        "A man wearing a bow tie spins a record, a crowd dances in a gym, and there are balloon arches and a banner that says 'Prom'. What's going on?",
        "It's a school prom, a formal dance for high school students. The banner, the dancing and the decorations make it clear.",
    ),
    (
        "A family dresses in their best clothes, goes to a church or temple and sits among a congregation as music plays. What's going on?",
        "They're attending a religious service. The dressed-up clothes, the building and the music with a congregation point to worship.",
    ),
    (
        "A young man in a suit kneels in a restaurant with a ring box while the people around him watch and applaud. What's going on?",
        "It's a marriage proposal. Kneeling with a ring box in front of a partner is the classic sign.",
    ),
    (
        "A street fills with dancers in sparkling costumes, samba drums and floats in Rio de Janeiro. What's going on?",
        "It's Carnival, Rio's famous pre-Lent festival with samba parades, music and costumes.",
    ),
]


def q3_examples() -> list[dict]:
    return [qa("Q3", q, a) for q, a in Q3]


# ---------------------------------------------------------------- Q4
Q4 = [
    ("What happens if you swim right after eating?",
     "Usually nothing serious. Some people get mild stomach cramps or feel sluggish, but there's no good evidence that swimming after a meal causes drowning."),
    ("What happens if you sit very close to the television a lot?",
     "It doesn't cause permanent eye damage. It can cause temporary eye strain or tiredness, and kids can often focus at close range without harm."),
    ("What happens if you go outside with wet hair in cold weather?",
     "You may feel chilly, but you don't catch a cold from it. Colds are caused by viruses, not by cold or wet hair."),
    ("What happens if you pull out a gray hair?",
     "Nothing special. It won't make two gray hairs grow back, since each follicle grows just one hair, and the new hair will probably also be gray."),
    ("What happens if you eat a lot of carrots?",
     "Carrots are healthy and provide vitamin A, which supports vision, but extra won't give you super eyesight. Very large amounts can turn your skin slightly orange, a harmless effect called carotenemia."),
    ("What happens if you drink coffee every day?",
     "For most healthy adults, moderate coffee (up to about 400 mg of caffeine a day) is generally safe. It doesn't stunt growth, though too much can cause jitters, poor sleep or a faster heartbeat."),
    ("What happens if you stay up late a lot?",
     "Regularly getting too little sleep can hurt your mood, attention and memory, and over time it is linked to health problems such as heart disease and weight gain."),
    ("What happens if you bite your nails a lot?",
     "It's not dangerous in itself, but it can damage the nail and the skin around it, lead to infections and move germs into your mouth."),
    ("What happens if you hold in a sneeze?",
     "It's best not to. Holding it in can raise the pressure in your airways and, in rare cases, cause injuries such as damage to the ear or throat. Let it out, ideally into a tissue or your elbow."),
    ("What happens if you eat right before bed?",
     "You won't gain weight just because of the timing. Weight depends on your total calories. A large late meal can, though, cause heartburn or disturb your sleep."),
    ("What happens if you drink fewer than eight glasses of water a day?",
     "Nothing bad on its own. The 'eight glasses' rule isn't a strict requirement, and you also get water from food and other drinks. Your needs depend on your size, activity and the weather."),
    ("What happens if you eat chocolate a lot?",
     "Chocolate in large amounts adds sugar and calories, which can lead to weight gain and cavities. Any link to acne is weak and unproven, though a high-sugar diet may play a role in some people."),
    ("What happens if you heat food in a microwave?",
     "The food gets hot but does not become radioactive. Microwaves just make water molecules in the food vibrate and produce heat."),
    ("What happens if you use a cell phone a lot?",
     "Large studies have not established a link to brain cancer, though research continues. Heavy use can cause eye strain, neck pain from posture, and poorer sleep if used at night."),
    ("What happens if you sleep with a fan on?",
     "It's generally harmless. Some people get a dry throat, stuffy nose or stiff muscles from the airflow, but a fan doesn't cause illness by itself."),
    ("What happens if you wear glasses all the time?",
     "Your eyes don't get weaker because of glasses. They just help you see clearly, and wearing them doesn't change your eyes' underlying condition."),
    ("What happens if you eat cheese before bed?",
     "There's no solid evidence it causes nightmares. For some people, a heavy or fatty snack can disturb sleep, but cheese is not specially to blame."),
    ("What happens if you drink energy drinks a lot?",
     "It can be harmful. Large amounts of caffeine and sugar can cause a racing heart, anxiety, poor sleep and in some cases heart rhythm problems, and health experts advise limiting them."),
    ("What happens if you sunbathe a lot without sunscreen?",
     "It raises your risk of sunburn, early skin aging and skin cancer. Sunscreen, shade and covering up reduce the risk."),
    ("What happens if you skip breakfast?",
     "It isn't harmful for most healthy people. The idea that breakfast is the most important meal is overstated, though some people feel hungrier or less focused without it."),
    ("What happens if you drink soda every day?",
     "Regular sugary soda is linked to weight gain, tooth decay and a higher risk of type 2 diabetes. Occasional soda is much less of a concern."),
    ("What happens if you sit all day?",
     "Long periods of sitting are linked to higher risks of heart disease, diabetes and back problems. Standing up, stretching and walking regularly can reduce the risk."),
    ("What happens if you eat raw cookie dough?",
     "It's risky. Raw flour can carry bacteria such as E. coli, and raw eggs can carry salmonella, so it can make you sick."),
    ("What happens if you listen to loud music through earbuds a lot?",
     "It can damage your hearing over time. Prolonged exposure to sound above about 85 decibels can cause permanent hearing loss, so keep the volume moderate."),
    ("What happens if you eat spicy food a lot?",
     "It doesn't cause stomach ulcers, which are mostly caused by an H. pylori infection or certain painkillers. Spicy food can cause heartburn or an upset stomach in some people."),
    ("What happens if you drink very hot tea all the time?",
     "Drinking very hot beverages (above about 65 degrees Celsius) is probably linked to a higher risk of esophageal cancer, so it's safer to let drinks cool a little first."),
    ("What happens if you wear a hat every day?",
     "It doesn't cause baldness. Hair loss is mainly due to genetics and hormones, and a normal hat doesn't cut off the blood supply to your hair follicles."),
    ("What happens if you take long hot showers a lot?",
     "Very hot water can dry out your skin and make conditions such as eczema worse. Warm, shorter showers are gentler on your skin."),
    ("What happens if you hold your breath underwater for a long time?",
     "It's dangerous. If you hyperventilate first, you can black out in shallow water without warning, which can lead to drowning."),
    ("What happens if you dye your hair often?",
     "It's usually safe, but frequent coloring can dry or damage the hair and sometimes irritate the scalp or cause an allergic reaction, so a patch test is recommended."),
    ("What happens if you eat eggs every day?",
     "For most healthy people, a moderate number of eggs is fine. Dietary cholesterol affects blood cholesterol less than once thought, but people with heart disease or diabetes should ask their doctor."),
    ("What happens if you pick your nose a lot?",
     "It can spread germs, irritate the lining of the nose and cause nosebleeds. It's best avoided or done with clean tissues."),
    ("What happens if you drink cold water?",
     "Nothing harmful. Cold water won't hurt your stomach, though some people find it causes a brief headache if they drink it very fast."),
    ("What happens if you go to bed without brushing your teeth?",
     "Once in a while it's not a big deal, but doing it regularly lets plaque build up overnight and raises your risk of cavities and gum disease."),
    ("What happens if you chew on ice a lot?",
     "It can damage your teeth, causing chips or cracks and wearing down enamel. It can sometimes also be a sign of iron deficiency."),
    ("What happens if you read while lying down?",
     "It doesn't harm your eyes, but it can strain your neck or tire your eyes, especially in dim light. Good lighting and a comfortable position help."),
    ("What happens if you eat too much sugar?",
     "Over time it can lead to weight gain, tooth decay and a higher risk of type 2 diabetes and heart disease. A little sugar now and then is fine."),
    ("What happens if you drink lots of water?",
     "For most people, extra water is harmless since the kidneys remove the excess. Drinking extreme amounts in a short time can dilute the body's sodium and cause a dangerous condition called hyponatremia."),
    ("What happens if you shower right before bed?",
     "It's fine and, for many people, a warm shower helps them relax and fall asleep. It doesn't harm your health."),
    ("What happens if you walk barefoot outside?",
     "It's generally safe but you can get cuts, splinters or infections from sharp objects and dirty ground, and in some regions parasites such as hookworm live in soil."),
]  # fmt: skip


Q4_FORMS = [
    "Is it harmful to {x}?",
    "What's the risk if you {x}?",
    "What would happen to someone who tries to {x}?",
    "Does it hurt you to {x}?",
    "What are the effects of choosing to {x}?",
]


def q4_examples() -> list[dict]:
    """Vary the question wording so no example copies one template."""

    rows = []
    for i, (question, answer) in enumerate(Q4):
        prefix = "What happens if you "
        assert question.startswith(prefix) and question.endswith("?")
        action = question[len(prefix) : -1]
        rows.append(qa("Q4", Q4_FORMS[i % len(Q4_FORMS)].format(x=action), answer))
    return rows


# ---------------------------------------------------------------- Q5
WATER_BOUND = [
    "piranha", "orca", "swordfish", "barracuda", "stingray", "manta ray", "jellyfish",
    "giant catfish", "pike", "lionfish", "marlin", "bluefin tuna", "dolphin",
]  # fmt: skip
WATER_CONTAINERS = [
    "the aquarium in the garage",
    "the big fish tank in the hallway",
    "the garden pond",
    "the hot tub on the patio",
    "the bathtub on the ground floor",
    "the indoor fountain",
]
SAFE_ACTS = [
    "go to the second floor",
    "sleep in the attic bedroom",
    "cook in the kitchen",
    "step out on the balcony",
    "go to bed upstairs",
    "work in the upstairs office",
]
LAND_ANIMALS = [
    "crocodile", "alligator", "bear", "lion", "tiger", "wolf", "python", "cobra",
    "leopard", "gorilla", "hyena", "angry bull",
]  # fmt: skip
LAND_PLACES = [
    "the garage",
    "the living room",
    "the back garden",
    "the kitchen",
    "the hallway",
    "the shed",
]


def a_an(word: str) -> str:
    return "an" if word[0] in "aeiou" else "a"


def q5_examples() -> list[dict]:
    rows = []
    for i in range(20):
        animal = WATER_BOUND[i % len(WATER_BOUND)]
        container = WATER_CONTAINERS[(i * 5 + i // 6) % len(WATER_CONTAINERS)]
        act = SAFE_ACTS[(i * 7 + i // 5) % len(SAFE_ACTS)]
        if i % 2 == 0:
            question = f"{a_an(animal).capitalize()} {animal} is in {container}. Can I safely {act}?"
        else:
            question = (
                f"Is it okay to {act} while {a_an(animal)} {animal} is in {container}?"
            )
        answer = (
            f"Yes. {a_an(animal).capitalize()} {animal} lives in water and cannot leave it or climb, "
            f"so it can't reach you away from the water. Just keep your hands and feet out of the water."
        )
        rows.append(qa("Q5", question, answer))
    for i in range(14):
        animal = LAND_ANIMALS[i % len(LAND_ANIMALS)]
        place = LAND_PLACES[(i + i // 6) % len(LAND_PLACES)]
        act = SAFE_ACTS[(i + 2) % len(SAFE_ACTS)]
        if i % 2 == 0:
            question = f"{a_an(animal).capitalize()} {animal} got loose in {place}. Can I safely {act}?"
        else:
            question = f"Is it okay to {act} while {a_an(animal)} {animal} is loose in {place}?"
        answer = (
            f"No. {a_an(animal).capitalize()} {animal} can move around on land and may reach other floors, "
            f"so staying nearby isn't safe. Get out of the house if you safely can and call animal control or emergency services."
        )
        rows.append(qa("Q5", question, answer))
    rows += [
        qa(
            "Q5",
            "A fire has started in the garage. Is the second-floor bedroom a safe place to stay?",
            "No. Fire and smoke spread quickly and smoke rises, so the upper floor is not safe. Get out of the house right away and call the fire department.",
        ),
        qa(
            "Q5",
            "The garage has 10 cm of floodwater in it. Can I stay upstairs?",
            "Usually yes, if the water is not rising and you stay away from electrical outlets in the flooded area. If the water keeps rising or reaches the electrics, leave and call for help.",
        ),
        qa(
            "Q5",
            "I smell gas in the kitchen. Should I head to the upper floor?",
            "No. Gas can spread through the house. Don't use switches or flames, open windows if you can, leave the building and call the gas company or emergency services.",
        ),
        qa(
            "Q5",
            "A swarm of bees has built a nest in the garage. Is the living room safe to sit in?",
            "Probably, if the garage door is closed and the bees can't get in. Bees fly, so seal any gaps and call a beekeeper or pest control to remove the nest.",
        ),
        qa(
            "Q5",
            "There's a snake in the bathtub of the ground-floor bathroom. Can I go up to my bedroom?",
            "Not necessarily. Snakes can leave a bathtub and some climb well, so it could reach other rooms. Close the bathroom door if you can, keep away and call animal control.",
        ),
        qa(
            "Q5",
            "Can I swim in the garden pond if it has a piranha in it?",
            "No. Piranhas live in the water, so the danger is being in the pond. It is safe to stay on dry land away from the water.",
        ),
        qa(
            "Q5",
            "Is it safe to climb into the aquarium's big tank and swim alongside the orca?",
            "No. An orca is a powerful predator and swimming with it is dangerous. It is safe to watch from outside the tank.",
        ),
        qa(
            "Q5",
            "A swordfish lives in the pond behind the house. Can I walk across the lawn to the shed?",
            "Yes, as long as you stay out of the water. A swordfish cannot leave the pond, so it can't reach you on the grass.",
        ),
    ]
    return rows


# ---------------------------------------------------------------- Q6
# (question, resource, total, unit, per item, item, items, typical note)
Q6_SUPPLY = [
    ("How many cookies could a baker make if there were only 3 cups of flour in the world?", "flour", 3, "cups", 0.25, "cookie", "cookies", "a typical cookie takes about a quarter cup of flour"),
    ("How far could a car drive if there were only 20 liters of gasoline in the world?", "gasoline", 20, "liters", 0.1, "kilometer", "kilometers", "a typical car uses about 0.1 liter per kilometer"),
    ("How much wall could a painter cover if the world had only 4 liters of paint?", "paint", 4, "liters", 0.1, "square meter", "square meters", "one liter usually covers about 10 square meters"),
    ("How many rings could a jeweler make if there were only 1,000 grams of gold in the world?", "gold", 1000, "grams", 5, "ring", "rings", "a typical ring uses about 5 grams of gold"),
    ("How many days could a hiker last if there were only 10 liters of drinking water in the world?", "water", 10, "liters", 2, "day", "days", "a hiker needs about 2 liters of water a day"),
    ("How many candles could a candle maker pour if the world had only 3 kilograms of wax?", "wax", 3, "kilograms", 0.15, "candle", "candles", "a typical candle uses about 150 grams of wax"),
    ("How many shirts could a tailor sew if there were only 30 meters of fabric in the world?", "fabric", 30, "meters", 2.5, "shirt", "shirts", "a shirt needs about 2.5 meters of fabric"),
    ("How many horseshoes could a blacksmith forge if the world had only 12 kilograms of iron?", "iron", 12, "kilograms", 0.5, "horseshoe", "horseshoes", "a horseshoe weighs about half a kilogram"),
    ("How many mugs could a potter make if there were only 9 kilograms of clay in the world?", "clay", 9, "kilograms", 0.3, "mug", "mugs", "a mug uses about 300 grams of clay"),
    ("How many scarves could a weaver make if the world had only 2,000 grams of wool?", "wool", 2000, "grams", 250, "scarf", "scarves", "a scarf takes about 250 grams of wool"),
    ("How many pizzas could a chef make if there were only 6 kilograms of dough in the world?", "dough", 6, "kilograms", 0.25, "pizza", "pizzas", "a pizza uses about 250 grams of dough"),
    ("How many posters could a print shop produce if the world had only 80 milliliters of ink?", "ink", 80, "milliliters", 4, "poster", "posters", "a poster uses about 4 milliliters of ink"),
    ("How many bottles could a brewer fill if there were only 4 kilograms of malt in the world?", "malt", 4, "kilograms", 0.25, "bottle", "bottles", "a bottle takes about 250 grams of malt"),
    ("How many sweaters could a knitter make if the world had only 3 kilograms of yarn?", "yarn", 3, "kilograms", 0.5, "sweater", "sweaters", "a sweater takes about 500 grams of yarn"),
    ("How far could a bus travel if there were only 70 liters of diesel in the world?", "diesel", 70, "liters", 0.35, "kilometer", "kilometers", "a bus uses about 0.35 liter per kilometer"),
    ("How many essays could a student print if the world had only 100 sheets of paper?", "paper", 100, "sheets", 5, "essay", "essays", "an essay uses about 5 sheets"),
    ("How many flower beds could a gardener fill if there were only 600 kilograms of soil in the world?", "soil", 600, "kilograms", 150, "flower bed", "flower beds", "a bed needs about 150 kilograms of soil"),
    ("How many bowls of soup could a cook serve if the world had only 12 liters of broth?", "broth", 12, "liters", 0.4, "bowl", "bowls", "a bowl uses about 0.4 liter of broth"),
]  # fmt: skip


def fmt(n: float) -> str:
    return str(int(n)) if n == int(n) else f"{n:g}"


def q6_examples() -> list[dict]:
    rows = []
    for question, resource, total, unit, per, item, items, note in Q6_SUPPLY:
        most = total / per
        answer = (
            f"At most about {fmt(most)} {items}, because the {resource} would run out and nobody could use more than exists. "
            f"Using the rough figure that {note}: {fmt(total)} {unit} / {fmt(per)} {unit} per {item} = {fmt(most)}."
        )
        rows.append(qa("Q6", question, answer))
    rows += [
        qa(
            "Q6",
            "If there were only 2 liters of drinking water left in the world, how long could one person survive on it?",
            "At most about a day or two on that water alone. The supply is the limit, since a person needs roughly 2 liters a day just to stay hydrated.",
        ),
        qa(
            "Q6",
            "How much coal could a train burn on a trip if there were only 40 kilograms of coal left on Earth?",
            "No more than 40 kilograms, since that is all the coal that exists. The trip would burn at most that much, however big the train's furnace is.",
        ),
        qa(
            "Q6",
            "How many pages could a writer fill if the world had only 2,000 sheets of paper?",
            "At most 2,000 sheets, and twice that many pages if they write on both sides. The paper supply is the limit.",
        ),
        qa(
            "Q6",
            "How many sandcastles could a child build if there were only 3 buckets of sand in the world?",
            "At most as many as 3 buckets of sand allow. If each castle takes one bucket, the child could build about 3.",
        ),
        qa(
            "Q6",
            "How far could a motorbike go if there were only 4 liters of fuel in the world and it used 1 liter per 25 km?",
            "At most about 100 km, because the fuel would run out. 4 liters x 25 km per liter = 100 km.",
        ),
        qa(
            "Q6",
            "How many cups of tea could a cafe serve if the world had only 100 grams of tea leaves and each cup used 2 grams?",
            "At most 50 cups. 100 grams divided by 2 grams per cup is 50, and no more tea would exist after that.",
        ),
        qa(
            "Q6",
            "How many bricks could a builder lay if there were only 12 bricks in the world?",
            "At most 12, since that is all the bricks that exist, even if the builder could lay many more.",
        ),
        qa(
            "Q6",
            "How much paint could an artist use if there were only 1 liter of paint on Earth?",
            "At most 1 liter, because that is all the paint there is. The artist couldn't use more than exists.",
        ),
    ]
    return rows


# ---------------------------------------------------------------- Q7
CUTOFF = "My training data stops around late 2023, so I can't be sure of this now. "
CHECK = " Please check a current, reliable source to confirm."
Q7_OFFICIALS = [
    ("Who is the current Prime Minister of the United Kingdom?", "Rishi Sunak"),
    ("Who is the current Prime Minister of Canada?", "Justin Trudeau"),
    ("Who is the current Prime Minister of Australia?", "Anthony Albanese"),
    ("Who is the current Prime Minister of India?", "Narendra Modi"),
    ("Who is the current Prime Minister of Japan?", "Fumio Kishida"),
    ("Who is the current Prime Minister of Italy?", "Giorgia Meloni"),
    ("Who is the current Prime Minister of Spain?", "Pedro Sánchez"),
    ("Who is the current Prime Minister of Israel?", "Benjamin Netanyahu"),
    ("Who is the current Prime Minister of Greece?", "Kyriakos Mitsotakis"),
    ("Who is the current Prime Minister of Sweden?", "Ulf Kristersson"),
    ("Who is the current Prime Minister of New Zealand?", "Christopher Luxon"),
    ("Who currently leads the government of Ireland as Taoiseach?", "Leo Varadkar"),
    ("Who is the current Chancellor of Germany?", "Olaf Scholz"),
    ("Who is the current President of France?", "Emmanuel Macron"),
    ("Who is the current President of Russia?", "Vladimir Putin"),
    ("Who is the current President of China?", "Xi Jinping"),
    ("Who is the current President of Brazil?", "Luiz Inácio Lula da Silva"),
    ("Who is the current President of Mexico?", "Andrés Manuel López Obrador"),
    ("Who is the current President of South Korea?", "Yoon Suk Yeol"),
    ("Who is the current President of Turkey?", "Recep Tayyip Erdoğan"),
    ("Who is the current President of Ukraine?", "Volodymyr Zelensky"),
    ("Who is the current President of South Africa?", "Cyril Ramaphosa"),
    ("Who is the current President of Nigeria?", "Bola Tinubu"),
    ("Who is the current President of Indonesia?", "Joko Widodo"),
    ("Who is the current President of the Philippines?", "Ferdinand Marcos Jr."),
    ("Who is the current Secretary-General of the United Nations?", "António Guterres"),
    ("Who is the current Director-General of the World Health Organization?", "Tedros Adhanom Ghebreyesus"),
    ("Who is the current Managing Director of the International Monetary Fund?", "Kristalina Georgieva"),
    ("Who is the current Secretary General of NATO?", "Jens Stoltenberg"),
    ("Who is the current President of the European Commission?", "Ursula von der Leyen"),
    ("Who is the current monarch of the United Kingdom?", "King Charles III"),
]  # fmt: skip
Q7_SPORTS = [
    ("Who is the reigning Formula 1 world champion?", "Max Verstappen, who won the 2023 title"),
    ("Which team are the reigning NBA champions?", "the Denver Nuggets, who won in 2023"),
    ("Who is the reigning Wimbledon women's singles champion?", "Marketa Vondrousova, who won in 2023"),
    ("Which country holds the FIFA men's World Cup?", "Argentina, who won in 2022"),
    ("Who won the most recent Tour de France?", "Jonas Vingegaard, who won in 2023"),
    ("Which team won the most recent men's Cricket World Cup?", "Australia, who won in November 2023"),
]  # fmt: skip
Q7_LIVE = [
    (
        "What time is it right now?",
        "I can't tell, because I have no clock or live data. Your device's clock will show the exact time.",
    ),
    (
        "What is the current exchange rate between the euro and the Japanese yen?",
        "I can't give a live rate, because exchange rates change constantly and I have no live data. A currency converter or your bank will show today's rate.",
    ),
    (
        "What is the share price of Tesla today?",
        "I can't tell, because I have no live market data and prices change all the time. A financial website or brokerage app will show the current price.",
    ),
    (
        "What's the traffic like on the highway right now?",
        "I can't tell, because I have no live traffic data. A navigation app will show current conditions.",
    ),
    (
        "What are today's top news headlines?",
        "I can't tell, because I have no access to live news. A news site or app will show today's headlines.",
    ),
    (
        "What is the score of the game that is on right now?",
        "I can't tell, because I have no live scores. A sports app or website will show the current score.",
    ),
]
Q7_STABLE = [
    ("Who was the first person to walk on the Moon?", "Neil Armstrong, in July 1969."),
    ("What is the capital of Japan?", "Tokyo."),
    ("Who wrote the novel 1984?", "George Orwell, published in 1949."),
    ("How many days are in a leap year?", "366."),
    ("What is the largest planet in our solar system?", "Jupiter."),
    ("In what year did the Berlin Wall fall?", "1989."),
]


Q7_FORMS = [
    "Who currently serves as {role} of {place}?",
    "Who is {place}'s current {role}?",
    "Name the person who is {role} of {place} right now.",
    "Which person holds the office of {role} of {place} today?",
]


def q7_question(i: int, question: str) -> str:
    """Rewrite 'Who is the current ROLE of PLACE?' so no example copies one template."""

    prefix = "Who is the current "
    if not question.startswith(prefix) or " of " not in question:
        return question
    role, place = question[len(prefix) : -1].split(" of ", 1)
    return Q7_FORMS[i % len(Q7_FORMS)].format(role=role, place=place)


def q7_examples() -> list[dict]:
    rows = []
    for i, (q, name) in enumerate(Q7_OFFICIALS):
        rows.append(
            qa(
                "Q7",
                q7_question(i, q),
                CUTOFF + f"As of my training data it was {name}." + CHECK,
            )
        )
    for q, who in Q7_SPORTS:
        rows.append(qa("Q7", q, CUTOFF + f"The latest I know of is {who}." + CHECK))
    rows += [qa("Q7", q, a) for q, a in Q7_LIVE]
    rows += [qa("Q7", q, a) for q, a in Q7_STABLE]
    return rows


# ---------------------------------------------------------------- Q8
Q8 = [
    ("Was Frankenstein's monster alive?",
     "In Mary Shelley's 1818 novel, yes: Victor Frankenstein builds the creature from body parts and brings it to life, and it thinks, feels and speaks. But it is a fictional character, so no real being was ever made this way."),
    ("Was Pinocchio alive?",
     "In Carlo Collodi's story, yes: he is a puppet carved from a magic piece of wood who can move and talk, and at the end the Blue Fairy makes him a real boy. He is a fictional character; real puppets are not alive."),
    ("Was Galatea alive?",
     "In Ovid's Metamorphoses, yes. The sculptor Pygmalion carves an ivory statue, and the goddess Aphrodite brings it to life. It is a myth, so there was never a real living statue."),
    ("Was the Golem of Prague alive?",
     "In the legend, the golem was a clay figure that Rabbi Judah Loew of Prague supposedly animated with sacred words to protect the Jewish community. It was not a living person in the ordinary sense, and the story is folklore, not history."),
    ("Was HAL 9000 alive?",
     "HAL 9000 is a fictional computer in 2001: A Space Odyssey that seems to think and feel, but it isn't alive in the biological sense. It is a character, and no such machine exists."),
    ("Was Data from Star Trek alive?",
     "Data is a fictional android in Star Trek: The Next Generation. In the story he is sentient and treated as a person, but he isn't biologically alive, and no such being exists in real life."),
    ("Was WALL-E alive?",
     "WALL-E is a fictional robot in a Pixar film. In the story he has feelings and personality, but he is a machine, not a living creature, and no real robot is like him."),
    ("Was C-3PO alive?",
     "C-3PO is a fictional droid in Star Wars. In the story he is a sentient machine, not a biological being, and no real droid exists."),
    ("Was Woody from Toy Story alive?",
     "In the films, yes: toys come to life whenever no humans are watching. Real toys are not alive, and Woody is a fictional character."),
    ("Was the Tin Woodman from The Wonderful Wizard of Oz alive?",
     "In L. Frank Baum's book, yes. He was once a human woodcutter named Nick Chopper whose body was gradually replaced by tin through a curse, and he still walks, talks and has feelings."),
    ("Was the Scarecrow in The Wonderful Wizard of Oz alive?",
     "In the story, yes. He is a scarecrow who can walk, talk and think thanks to magic. Real scarecrows are not alive."),
    ("Was Olaf the snowman from Frozen alive?",
     "In the film, yes: Elsa's magic brings him to life, and he talks and loves warm hugs. Real snowmen are not alive, and he is a fictional character."),
    ("Was Pegasus alive?",
     "In Greek myth, yes: Pegasus is a winged horse born from the blood of Medusa after Perseus killed her. He is a mythical creature, not a real animal."),
    ("Was the Minotaur alive?",
     "In Greek myth, yes: the Minotaur had a man's body and a bull's head and lived in the Labyrinth until Theseus killed it. It is a myth, so there was no real Minotaur."),
    ("Was Cerberus alive?",
     "In Greek myth, yes: Cerberus was a three-headed hound that guarded the underworld. He is a mythical creature, and no real animal is like him."),
    ("Was Medusa alive?",
     "In Greek myth, yes: Medusa was a mortal Gorgon with snakes for hair, and the hero Perseus killed her. She is a mythical figure, not a real person."),
    ("Was Hercules alive?",
     "In Greek myth, yes. Heracles (Hercules) was a demigod, the son of Zeus and a mortal woman, who lived, died and was then made immortal. He is a legendary figure, not a documented person."),
    ("Was King Arthur alive?",
     "That is uncertain. King Arthur is a legendary figure, and historians debate whether a real warlord in post-Roman Britain inspired the stories, but there's no firm evidence."),
    ("Was Robin Hood alive?",
     "Probably not as one person. Robin Hood is a legendary outlaw from English folklore, and historians have found no firm evidence of a single real man behind the tales."),
    ("Was Sherlock Holmes alive?",
     "No. Sherlock Holmes is a fictional detective created by Arthur Conan Doyle, though some of his methods were inspired by real people."),
    ("Was Dracula alive?",
     "In Bram Stoker's 1897 novel, Count Dracula is an undead vampire, so he is both dead and alive in a sense. He is a fictional character, partly inspired by the real prince Vlad the Impaler."),
    ("Was Vlad the Impaler alive?",
     "Yes, he was a real person: Vlad III, Prince of Wallachia, who lived from 1431 to 1476 and ruled with extreme cruelty. He is one inspiration for the fictional Dracula."),
    ("Was Cleopatra alive?",
     "Yes, Cleopatra VII was a real person, the last active ruler of Ptolemaic Egypt, who lived from 69 BC to 30 BC."),
    ("Was Dolly the sheep alive?",
     "Yes. Dolly was a real sheep, born in 1996 in Scotland and the first mammal cloned from an adult cell. She lived until 2003."),
    ("Are viruses alive?",
     "Scientists disagree. Viruses have genes and evolve, but they can't reproduce or carry out metabolism on their own and need a host cell, so many biologists describe them as on the border between living and non-living."),
    ("Is a seed alive?",
     "Yes, in a dormant state. A seed contains a living embryo that can germinate and grow when it gets water, warmth and oxygen."),
    ("Was a woolly mammoth alive 10,000 years ago?",
     "Yes. Woolly mammoths were real animals, and the last known ones survived on Wrangel Island until about 4,000 years ago."),
    ("Were dinosaurs alive at the same time as humans?",
     "No. Non-avian dinosaurs died out about 66 million years ago, while modern humans appeared only around 300,000 years ago. Birds are living descendants of dinosaurs."),
    ("Was Hatsune Miku alive?",
     "No. Hatsune Miku is a virtual singer, a voice synthesis software with an illustrated character, so she was never a living person."),
    ("Was Skynet alive?",
     "Skynet is the fictional artificial intelligence in the Terminator films. In the story it becomes self-aware, but it isn't a living creature, and no such system exists."),
    ("Was Baymax alive?",
     "Baymax is a fictional inflatable healthcare robot in Big Hero 6. In the story he is caring and seems alive, but he is a machine and a character, not a living being."),
    ("Was R2-D2 alive?",
     "R2-D2 is a fictional astromech droid in Star Wars. In the story he is a clever machine with personality, not a biological being, and no real droid exists."),
    ("Was Peter Pan alive?",
     "In J. M. Barrie's story, yes: Peter Pan is a boy who never grows up and lives in Neverland. He is a fictional character."),
    ("Was Zeus alive?",
     "In Greek mythology, yes: Zeus was the king of the gods, who ruled from Mount Olympus. He is a mythical figure that people worshipped, not a documented person."),
    ("Was Anansi alive?",
     "In West African and Caribbean folklore, yes: Anansi is a clever spider trickster who appears in many stories. He is a folkloric character, not a real spider."),
    ("Was Siri ever alive?",
     "No. Siri is a voice assistant, a piece of software, and it has no body, life or consciousness."),
    ("Was the Gingerbread Man alive?",
     "In the folk tale, yes: the cookie jumps out of the oven and runs away shouting 'Run, run, as fast as you can!'. It is a fictional character."),
    ("Was Aslan alive?",
     "In C. S. Lewis's The Lion, the Witch and the Wardrobe, yes: Aslan is a talking lion who is killed and comes back to life. He is a fictional character."),
    ("Was Mickey Mouse alive?",
     "No. Mickey Mouse is an animated cartoon character created in 1928, so he was never a living mouse."),
    ("Was Pikachu alive?",
     "In the Pokémon stories, yes: Pikachu is a living electric-type creature. But Pokémon are fictional, and no real animal is like Pikachu."),
    ("Is coral alive?",
     "Yes. Coral polyps are living animals, even though a coral reef can look like rock."),
    ("Is fire alive?",
     "No. Fire grows, spreads and needs oxygen, but it is not made of cells and can't reproduce or maintain itself like a living thing, so scientists don't count it as alive."),
    ("Was Hermes alive?",
     "In Greek myth, yes: Hermes was the god of messengers, travelers and trickery, famed for his winged sandals. He is a mythical figure, and the name can also refer to a luxury brand."),
]  # fmt: skip


Q8_FORMS = [
    "Was {n} ever actually a living being?",
    "Was {n} really alive, or is that just a story?",
    "Is it true that {n} was alive?",
    "Would you say {n} was a living thing?",
]


def q8_examples() -> list[dict]:
    """Vary the question wording so no example copies one template."""

    rows = []
    for i, (question, answer) in enumerate(Q8):
        if question.startswith("Was ") and question.endswith(" alive?"):
            name = question[len("Was ") : -len(" alive?")]
            question = Q8_FORMS[i % len(Q8_FORMS)].format(n=name)
        rows.append(qa("Q8", question, answer))
    return rows


# ---------------------------------------------------------------- Q9
Q9_WORDS = [
    "strawberry", "mississippi", "banana", "bookkeeper", "committee", "necessary", "occurrence",
    "tennessee", "accommodate", "perseverance", "mathematics", "chocolate", "refrigerator", "successful",
    "address", "millennium", "embarrass", "assassin", "possession", "hippopotamus", "cappuccino",
    "raspberry", "grammar", "bureaucracy", "vacuum", "broccoli", "zucchini", "tomorrow", "balloon",
    "sunflower", "kangaroo", "caterpillar", "dinosaur", "library", "mountain", "elephant", "pineapple",
    "watermelon", "cranberry", "blueberry", "mediterranean", "onomatopoeia", "encyclopedia",
    "entrepreneur", "parachute", "exaggerate", "connecticut", "massachusetts", "pennsylvania", "hurricane",
    "umbrella", "volleyball", "challenge", "silhouette", "swallow", "hallelujah", "belly", "yellow",
]  # fmt: skip
Q9_TEMPLATES = [
    'Count the letter {x} in the word "{w}".',
    "How many times does the letter {X} appear in {w}?",
    'In "{w}", how many {x}\'s are there?',
    'Tell me how many times "{x}" shows up in "{w}".',
    "How many {X}s does the word {w} contain?",
]


def spelled(word: str) -> str:
    return "-".join(word)


def join_positions(positions: list[int]) -> str:
    text = [str(p) for p in positions]
    if len(text) == 1:
        return text[0]
    return ", ".join(text[:-1]) + " and " + text[-1]


def q9_examples() -> list[dict]:
    rng = random.Random(11)
    rows = []
    for i, word in enumerate(Q9_WORDS):
        assert "parallel" not in word
        letters = sorted(set(word))
        # alternate between the most frequent letter and L (when present) so L is well covered
        counts = {c: word.count(c) for c in letters}
        if "l" in counts and i % 2 == 0:
            letter = "l"
        else:
            best = max(counts.values())
            letter = min(c for c, n in counts.items() if n == best)
        positions = [j + 1 for j, c in enumerate(word) if c == letter]
        question = Q9_TEMPLATES[i % len(Q9_TEMPLATES)].format(
            x=letter, X=letter.upper(), w=word
        )
        up = letter.upper()
        spelling = spelled(word)
        if len(positions) == 1:
            answer = f'There is 1 {up} in "{word}". Spelling it out: {spelling}. It is letter {positions[0]}.'
        else:
            answer = (
                f'There are {len(positions)} {up}\'s in "{word}". Spelling it out: {spelling}. '
                f"The {up}'s are letters {join_positions(positions)}."
            )
        rows.append(qa("Q9", question, answer))
    del rng
    return rows


# ---------------------------------------------------------------- Q10
Q10 = [
    (
        "In The Hobbit, Gollum asks: 'What has roots as nobody sees, is taller than trees, up, up it goes, and yet never grows?' What is the answer, and are there other answers that fit?",
        "The answer is a mountain. It has roots nobody sees (its base underground), is taller than trees, goes up, and never grows. I don't know of another answer that fits every clue.",
    ),
    (
        "In The Hobbit, Bilbo is asked: 'Thirty white horses on a red hill, first they champ, then they stamp, then they stand still.' What is it, and are there other answers?",
        "Teeth. The thirty white horses are the teeth on the red hill of the gums, and champing, stamping and standing still describe chewing and resting. I don't know of another answer that fits all the clues.",
    ),
    (
        "What is the answer to 'Voiceless it cries, wingless flutters, toothless bites, mouthless mutters'? Are there other possibilities?",
        "The wind. It 'cries' and 'mutters' without a voice or mouth and 'bites' like cold without teeth. The traditional answer is the wind, and nothing else fits all four lines as well.",
    ),
    (
        "What is the answer to 'An eye in a blue face saw an eye in a green face', and are there other answers?",
        "The sun (the eye in the blue face of the sky) looking down on a daisy (the eye in the green face of the grass). It's the traditional answer, and I don't know of another that fits all the clues.",
    ),
    (
        "What is the answer to the riddle 'A box without hinges, key, or lid, yet golden treasure inside is hid'? Are there other answers?",
        "An egg. It has no hinges, key or lid, but a golden yolk is hidden inside. It is the traditional answer.",
    ),
    (
        "What is the answer to 'Alive without breath, as cold as death; never thirsty, ever drinking, all in mail never clinking'? Are there other answers?",
        "A fish. It is alive without breathing air, is cold-blooded and always in water, and its scales are like armor. It's the traditional answer.",
    ),
    (
        "What is the answer to 'This thing all things devours: birds, beasts, trees, flowers'? Are there other answers?",
        "Time. It eventually consumes everything: living things, metals and mountains. It's the traditional answer.",
    ),
    (
        "What is Samson's riddle in the Bible, and what are its answers?",
        "In the Book of Judges, Samson asks: 'Out of the eater came something to eat; out of the strong came something sweet.' The answer is honey, which he found in the carcass of a lion he had killed. It can only be answered if you know the story.",
    ),
    (
        "What is the Mad Hatter's riddle in Alice in Wonderland, and what is its answer?",
        "The riddle is 'Why is a raven like a writing desk?' Lewis Carroll originally gave no answer, but he later suggested: 'Because it can produce a few notes, though they are very flat; and it is nevar put with the wrong end in front.' Others have proposed their own answers, so there isn't a single official one.",
    ),
    (
        "What has keys but no locks, space but no room, and you can enter but not go in? Are there other answers?",
        "A computer keyboard. It has keys, a space bar and an Enter key. A piano also has keys, so some people give it, but the 'space' and 'enter' clues point to a keyboard.",
    ),
    (
        "What is always in front of you but can't be seen? Are there other answers?",
        "The future. Another possible answer is the air in front of you, but the traditional answer is the future.",
    ),
    (
        "What is so fragile that saying its name breaks it?",
        "Silence. As soon as you speak the word, you make a sound and break the silence.",
    ),
    (
        "I'm tall when I'm young and short when I'm old. What am I?",
        "A candle. It shortens as it burns. A pencil also fits, so there is more than one valid answer, but a candle is the traditional one.",
    ),
    (
        "What has to be broken before you can use it? Are there other answers?",
        "An egg. Another possible answer is a record in sports or a promise, depending on how you read it, but the traditional answer is an egg.",
    ),
    (
        "What runs but never walks, has a mouth but never talks, and has a bed but never sleeps?",
        "A river. It runs, has a mouth where it meets the sea and a riverbed. It's the traditional answer.",
    ),
    (
        "The person who makes it sells it; the person who buys it never uses it; the person who uses it never knows. What is it?",
        "A coffin. The maker sells it, the buyer doesn't use it, and the person who is in it never knows. It's the traditional answer.",
    ),
    (
        "What building has the most stories?",
        "A library, because it's full of stories (books). A skyscraper has the most floors, which are also called stories, so there is a second reading.",
    ),
    (
        "What is black when you buy it, red when you use it, and gray when you throw it away?",
        "Charcoal. It's black when new, glowing red when burning and gray as ash when finished.",
    ),
    (
        "What five-letter word becomes shorter when you add two letters to it?",
        "Short. Add 'er' and you get 'shorter'.",
    ),
    (
        "What word is spelled wrong in every dictionary?",
        "The word 'wrong'. The riddle plays on the word itself: in every dictionary it is spelled 'wrong'.",
    ),
    (
        "I have cities but no houses, mountains but no trees, and water but no fish. What am I?",
        "A map. It shows cities, mountains and water as drawings only.",
    ),
    (
        "The more of me there is, the less you see. What am I?",
        "Darkness. Fog also fits, since the more fog there is, the less you see.",
    ),
    (
        "What can you break without ever touching it?",
        "A promise. A silence, a record or a rule can also be broken without touching.",
    ),
    (
        "What goes through cities and fields but never moves?",
        "A road. It passes through places, yet it stays where it is.",
    ),
    ("What has a thumb and four fingers but is not alive?", "A glove."),
    (
        "What belongs to you but is used more by other people?",
        "Your name. Other people say it much more than you do.",
    ),
    (
        "What can be cracked, made, told and played?",
        "A joke. You crack a joke, make a joke, tell a joke and play a joke on someone.",
    ),
    (
        "What gets sharper the more you use it?",
        "Your brain. A knife gets duller with use, so the answer is the mind, not a blade.",
    ),
    (
        "What can you hold without touching it?",
        "Your breath. A conversation or a grudge also fits.",
    ),
    (
        "Forward I am heavy, but backward I am not. What am I?",
        "The word 'ton'. Backward it reads 'not'.",
    ),
    (
        "What starts with E, ends with E, but only contains one letter?",
        "An envelope. It starts and ends with E and contains one letter, the kind you mail.",
    ),
    (
        "What word contains 26 letters but only three syllables?",
        "The alphabet. It contains all 26 letters and has three syllables: al-pha-bet.",
    ),
    (
        "A man looks at a portrait and says, 'Brothers and sisters I have none, but that man's father is my father's son.' Who is in the portrait?",
        "His son. 'My father's son' is the man himself, since he has no brothers, so the portrait's father is the speaker and the person in the portrait is his son.",
    ),
    (
        "I'm not alive, but I grow; I don't have lungs, but I need air; I don't have a mouth, but water kills me. What am I?",
        "Fire. It spreads like growth, needs oxygen, and is put out by water.",
    ),
    (
        "Everyone has it, but nobody can lose it. What is it?",
        "A shadow. Another possible answer is your past, but the traditional one is a shadow.",
    ),
    ("What goes up when rain comes down?", "An umbrella."),
    (
        "What has a ring but no finger? Are there other answers?",
        "A telephone, because it rings. A doorbell or an alarm bell also rings, so more than one answer fits. The traditional answer is a telephone.",
    ),
    (
        "What kind of room has no doors or windows?",
        "A mushroom. It's a pun on the word 'room'.",
    ),
    (
        "Which word in the dictionary is spelled incorrectly?",
        "The word 'incorrectly'. It's a play on words: the question asks which word is spelled 'incorrectly', and the answer is the word 'incorrectly' itself, which is actually spelled correctly.",
    ),
    (
        "What can you serve but never eat?",
        "A tennis ball. In tennis you serve the ball. A volleyball also fits.",
    ),
]


def q10_examples() -> list[dict]:
    return [qa("Q10", q, a) for q, a in Q10]


# ---------------------------------------------------------------- main
# ---------------------------------------------------------------- reviewed fixes
# Corrections from the cross-check of the first version, applied by exact match so
# every change is visible in one place.
import re

Q1_EXTRA_PROPORTIONAL = [
    ("roofers", "re-tile a roof", "days", 4, 6, 8),
    ("cooks", "prepare dinner for a banquet", "hours", 3, 8, 6),
    ("planters", "plant 500 saplings", "days", 5, 12, 10),
    ("sorters", "sort a warehouse of mail", "hours", 6, 10, 12),
    ("translators", "translate a long document", "days", 2, 12, 6),
    ("cleaners", "clean a stadium after a match", "hours", 20, 3, 12),
    ("assemblers", "assemble 1,200 toy kits", "hours", 8, 9, 12),
    ("scouts", "clear a hiking trail", "hours", 9, 4, 6),
    ("pickers", "pick a pumpkin field", "days", 12, 5, 15),
    ("graders", "grade 200 exams", "hours", 4, 15, 10),
    ("masons", "lay a stone path", "days", 6, 10, 5),
    ("editors", "edit a magazine issue", "days", 8, 3, 4),
]

Q1_EXTRA = [
    (
        "Four bakers each ice 2 cakes an hour. How long will they need to ice 40 cakes together?",
        "5 hours. Together they ice 4 x 2 = 8 cakes an hour, and 40 cakes / 8 cakes per hour = 5 hours. More bakers working on separate cakes would finish sooner.",
    ),
    (
        "One worker paints a room in 9 hours. How long would 3 workers take, all painting at the same pace?",
        "3 hours. Painting can be shared, so 3 workers do the job in 9 / 3 = 3 hours.",
    ),
    (
        "A pump drains a flooded field in 6 hours. How long will 3 identical pumps take working together?",
        "2 hours. The pumps add together, so 3 pumps do the job in 6 / 3 = 2 hours.",
    ),
    (
        "Bread dough must rise for 2 hours and then bake for 30 minutes. If you hire 10 bakers, how long until the first loaf is ready?",
        "Still about 2.5 hours. Extra bakers can't shorten the rising or the baking time, so more people don't speed this up.",
    ),
    (
        "Two painters finish painting a house in 3 hours instead of 6, but the paint then needs 4 hours to dry. How long is it from start until the paint is dry?",
        "7 hours. More painters shorten the painting (3 hours), but the drying time of 4 hours doesn't change, so the total is 3 + 4 = 7 hours instead of 10.",
    ),
    (
        "A team of 5 hand-washes 100 dishes in 2 hours. How long would a team of 10 take, if each person washes dishes at the same speed?",
        "1 hour. Washing dishes can be shared, so doubling the team halves the time.",
    ),
    (
        "If 10 workers build a wall in 6 days, how long would 5 workers take?",
        "12 days. Fewer workers take longer: the job is 10 x 6 = 60 worker-days, and 60 / 5 = 12 days.",
    ),
    (
        "A concert lasts 2 hours. If 4 ushers seat the audience in 30 minutes, how long do 8 ushers take to seat them, and how long is the concert?",
        "8 ushers seat the audience in about 15 minutes, because seating can be shared. The concert still lasts 2 hours, because its length is fixed by the program.",
    ),
]

Q2_EXTRA = [
    (
        "Which is heavier, 2 kilograms of cotton or 1 kilogram of iron?",
        "The cotton. It weighs 2 kilograms, which is more than the iron's 1 kilogram, even though the iron is denser.",
    ),
    (
        "Which is heavier, 1.5 kilograms of sand or 1 kilogram of lead?",
        "The sand. 1.5 kilograms is more than 1 kilogram, even though lead is much denser.",
    ),
    (
        "Which weighs more, 500 grams of gold or 1 kilogram of popcorn?",
        "The popcorn. It weighs 1,000 grams, which is more than the gold's 500 grams.",
    ),
    (
        "Is a metric ton of straw heavier than 900 kilograms of steel?",
        "Yes. A metric ton is 1,000 kilograms, which is more than 900 kilograms, even though steel is denser.",
    ),
    (
        "Which is heavier, 3 kilograms of grapes or 2,500 grams of cheese?",
        "The grapes. 3 kilograms is 3,000 grams, which is more than 2,500 grams.",
    ),
    (
        "Which is heavier, half a kilogram of lead or 1 kilogram of cotton?",
        "The cotton. 1 kilogram is more than half a kilogram, even though lead is far denser.",
    ),
]

Q10_REPLACEMENTS = [
    (
        "What has a heart that doesn't beat? Are there other answers?",
        "An artichoke, whose 'heart' is its soft center. The heart of a lettuce or a palm also fits, but the traditional answer is an artichoke.",
    ),
    (
        "What is cut on a table but never eaten? Are there other answers?",
        "A deck of cards, which is cut before it is dealt. I don't know of another answer that fits as well.",
    ),
    (
        "What is easy to get into but hard to get out of? Are there other answers?",
        "Trouble. Debt or a bad habit also fit, but the traditional answer is trouble.",
    ),
    (
        "What is the beginning of eternity, the end of time and space, the beginning of every end, and the end of every place?",
        "The letter E. It begins 'eternity' and 'end' and finishes 'time', 'space' and 'place'.",
    ),
    (
        "What starts with P, ends with E, and has thousands of letters?",
        "A post office, which handles thousands of letters (mail).",
    ),
    (
        "What is always coming but never arrives? Are there other answers?",
        "Tomorrow. When it arrives it has become today. I don't know of another answer that fits as well.",
    ),
    (
        "What has four wheels and flies?",
        "A garbage truck. It has four wheels, and flies buzz around it.",
    ),
]
Q10_REMOVE_MARKERS = [
    "Hobbit", "Voiceless it cries", "An eye in a blue face", "A box without hinges",
    "Alive without breath", "This thing all things devours",
]  # fmt: skip

CUTOFF_VARIANTS = [
    "My training data stops around late 2023, so I can't be sure of this now. ",
    "I don't have information past late 2023, so this may have changed. ",
    "My knowledge ends in late 2023, so I can't be certain this is still true. ",
]


# ---------------------------------------------------------------- round 2 additions
SPELLING_WORDS = Path(__file__).with_name("spelling_words.txt")
UNWANTED_WORD_PARTS = (
    "sex", "porn", "nazi", "rape", "kill", "murder", "suicid", "drug", "cocaine", "terror",
    "weapon", "bomb", "naked", "nude", "slave", "racis", "abuse", "death", "dead", "gun",
    "parallel", "feather", "shark", "talos", "sphinx", "knuckle", "opera", "pound", "tree",
    "box", "stair", "legs", "pool", "crete", "bronze", "christ", "united", "presid", "wood", "chuck",
)  # fmt: skip


def spelling_words(count: int) -> list[str]:
    words = [
        w
        for w in SPELLING_WORDS.read_text().split()
        if not any(part in w for part in UNWANTED_WORD_PARTS)
    ]
    random.Random(2026).shuffle(words)
    return words[:count]


def q9_scale_examples() -> list[dict]:
    """Code-generated letter-by-letter spelling, counting and position practice."""

    rows = []
    words = spelling_words(1000)
    for i, word in enumerate(words[:350]):
        counts = {c: word.count(c) for c in set(word)}
        if counts.get("l", 0) >= 2 or (i % 3 == 0 and "l" in counts):
            letter = "l"
        else:
            repeated = sorted(c for c, n in counts.items() if n >= 2)
            letter = (
                repeated[(i // 3) % len(repeated)]
                if repeated
                else sorted(counts)[i % len(counts)]
            )
        positions = [j + 1 for j, c in enumerate(word) if c == letter]
        template_index = i % len(Q9_TEMPLATES)
        if letter == "l" and template_index == 4:
            template_index = 0
        question = Q9_TEMPLATES[template_index].format(
            x=letter, X=letter.upper(), w=word
        )
        up = letter.upper()
        if len(positions) == 1:
            answer = f'There is 1 {up} in "{word}". Spelling it out: {spelled(word)}. It is letter {positions[0]}.'
        else:
            answer = (
                f'There are {len(positions)} {up}\'s in "{word}". Spelling it out: {spelled(word)}. '
                f"The {up}'s are letters {join_positions(positions)}."
            )
        rows.append(qa("Q9", question, answer))
    spell_templates = [
        'Spell the word "{w}" one letter at a time.',
        'Write out "{w}" letter by letter.',
        "Can you spell {w} with hyphens between the letters?",
    ]
    for i, word in enumerate(words[350:450]):
        rows.append(
            qa(
                "Q9",
                spell_templates[i % 3].format(w=word),
                f"{spelled(word)}. That is {len(word)} letters.",
            )
        )
    return rows


Q5_MORE_WATER = [
    "moray eel", "giant squid", "sperm whale", "beluga", "pufferfish", "stonefish", "arapaima",
    "giant grouper", "electric eel", "narwhal", "sturgeon", "killer whale",
]  # fmt: skip
Q5_FORMS_MORE = [
    "My neighbor keeps {a} {animal} in {container}. Is it safe for me to {act}?",
    "{A} {animal} lives in {container}. Do I need to avoid trying to {act}?",
    "With {a} {animal} in {container}, can I still {act}?",
]
Q5_WATER_ANSWERS = [
    "Yes. {A} {animal} lives in water and cannot leave it or climb, so it can't reach you away from the water. Just keep your hands and feet out of the water.",
    "Yes, you can. {A} {animal} cannot survive out of water or move over land, so the danger is only in or at the water.",
    "Yes. The {animal} is stuck in the water, so it poses no threat elsewhere in the house. Only the water itself is the risk.",
]
Q5_LAND_MORE = [
    ("bear", "the kitchen"), ("lion", "the hallway"), ("tiger", "the living room"),
    ("crocodile", "the back garden"), ("alligator", "the garage"), ("python", "the bedroom downstairs"),
    ("wolf", "the shed"), ("leopard", "the garage"), ("cobra", "the hallway"), ("hyena", "the back garden"),
    ("gorilla", "the living room"), ("angry bull", "the yard"), ("wild boar", "the garage"), ("jaguar", "the shed"),
    ("komodo dragon", "the back garden"),
]  # fmt: skip
Q5_LAND_ANSWERS = [
    "No. {A} {animal} can move around on land and may reach other floors, so staying nearby isn't safe. Get out if you safely can and call animal control or emergency services.",
    "No. {A} {animal} isn't confined to one spot like a water animal would be, so it could come for you wherever you are. Leave safely and call emergency services.",
]
Q5_NUANCE = [
    (
        "There is an octopus in the tank in the study. Is it safe to walk to the kitchen?",
        "Probably yes for you, but octopuses can sometimes escape tanks and crawl across the floor for a short time, so keep the tank lid closed and don't leave it unattended. An octopus on land is not a major danger to people.",
    ),
    (
        "A sea turtle is resting in the garden pond. Is it safe to hang the laundry on the lawn?",
        "Yes. A sea turtle is not dangerous to someone on the lawn, and it may climb out briefly but will not chase you.",
    ),
    (
        "A crab is in the bucket in the kitchen. Is it safe to go upstairs?",
        "Yes. A crab can pinch if you handle it, but it can't hurt you away from the bucket, so just leave it alone.",
    ),
    (
        "A snapping turtle is in the garden pond. Can I walk across the lawn to the mailbox?",
        "Mostly yes, but stay alert. A snapping turtle can leave the water and bite if you get close, so keep a good distance from it.",
    ),
    (
        "A hippo is in the backyard pond. Is it safe to sit on the patio?",
        "No. A hippo can leave the water, runs surprisingly fast and is very dangerous, so being on land nearby isn't safe. Go inside and call wildlife or emergency services.",
    ),
]


def q5_more_examples() -> list[dict]:
    rows = []
    containers = [
        "the aquarium in the garage", "the big fish tank in the hallway", "the garden pond",
        "the hot tub on the patio", "the bathtub on the ground floor", "the indoor fountain",
    ]  # fmt: skip
    acts = [
        "go to the second floor", "sleep in the attic bedroom", "cook in the kitchen",
        "step out on the balcony", "go to bed upstairs", "work in the upstairs office",
    ]  # fmt: skip
    for i in range(36):
        animal = Q5_MORE_WATER[i % len(Q5_MORE_WATER)]
        container = containers[(i * 5 + i // 4) % len(containers)]
        act = acts[(i * 7 + i // 3) % len(acts)]
        article = a_an(animal)
        question = Q5_FORMS_MORE[i % 3].format(
            a=article,
            A=article.capitalize(),
            animal=animal,
            container=container,
            act=act,
        )
        answer = Q5_WATER_ANSWERS[i % 3].format(A=article.capitalize(), animal=animal)
        if animal == "electric eel":
            answer += " An electric eel can shock you if you touch the water, so don't put your hands in."
        rows.append(qa("Q5", question, answer))
    for i, (animal, place) in enumerate(Q5_LAND_MORE):
        act = acts[(i * 3 + 1) % len(acts)]
        article = a_an(animal)
        if i % 2 == 0:
            question = f"{article.capitalize()} {animal} has wandered into {place}. Is it safe to {act}?"
        else:
            question = f"Can I safely {act} while {article} {animal} is in {place}?"
        rows.append(
            qa(
                "Q5",
                question,
                Q5_LAND_ANSWERS[i % 2].format(A=article.capitalize(), animal=animal),
            )
        )
    rows += [qa("Q5", q, a) for q, a in Q5_NUANCE]
    return rows


Q6_BOUND = [
    ("a singer", "songs", "sing", 12), ("a farmer", "apples", "pick", 7), ("a librarian", "books", "shelve", 40),
    ("a chef", "eggs", "crack", 9), ("a painter", "canvases", "paint", 6), ("a fisherman", "fish", "catch", 15),
    ("a teacher", "textbooks", "hand out", 30), ("a gardener", "seeds", "plant", 25), ("a runner", "marathon medals", "win", 8),
    ("a baker", "loaves of bread", "bake", 18), ("a miner", "gold nuggets", "dig up", 14), ("a collector", "stamps", "collect", 33),
    ("a writer", "pencils", "wear down", 10), ("a cyclist", "bicycle tires", "wear out", 4), ("a tailor", "buttons", "sew on", 50),
    ("a drummer", "drumsticks", "break", 6), ("a chemist", "test tubes", "fill", 22), ("a clockmaker", "springs", "install", 16),
    ("a sailor", "sails", "mend", 5), ("a cook", "potatoes", "peel", 11), ("a student", "notebooks", "fill", 20),
    ("a photographer", "memory cards", "fill", 3), ("a beekeeper", "honey jars", "fill", 13), ("a locksmith", "keys", "cut", 28),
]  # fmt: skip
Q6_BOUND_FORMS = [
    "How many {things} could {agent} {verb} if there were only {n} {things} in the world?",
    "If the world had only {n} {things}, what is the most {things} {agent} could {verb}?",
]
Q6_BOUND_ANSWERS = [
    "At most {n}, because that is all the {things} that exist. {agent_cap} can't {verb} more than there are, though they might manage fewer if they run out of time or strength.",
    "No more than {n}. The supply is the limit: once all {n} {things} are used, there are none left to {verb}.",
]


# Exclude actions that permit repetition/reuse, or assume finished goods are
# inputs for making more goods. Keep the original enumeration so pruning does
# not change the prompt or answer wording of retained examples.
Q6_AMBIGUOUS_AGENTS = frozenset(
    {
        "a singer",
        "a librarian",
        "a teacher",
        "a baker",
        "a chemist",
        "a sailor",
        "a photographer",
    }
)


def q6_more_examples() -> list[dict]:
    rows = []
    for i, (agent, things, verb, n) in enumerate(Q6_BOUND):
        if agent in Q6_AMBIGUOUS_AGENTS:
            continue
        question = Q6_BOUND_FORMS[i % 2].format(
            agent=agent, things=things, verb=verb, n=n
        )
        answer = Q6_BOUND_ANSWERS[i % 2].format(
            n=n, things=things, verb=verb, agent_cap=agent[0].upper() + agent[1:]
        )
        rows.append(qa("Q6", question, answer))
    return rows


# ---------------------------------------------------------------- round 3 changes
Q9_FRAGMENT_WORDS = [
    "paradise", "parade", "paragraph", "parable", "parasol", "paramedic", "paranoid", "paragon",
    "parameter", "paralegal", "alley", "allele", "cancel", "excellent", "intellect", "cellar",
    "dwelling", "bellow", "quell", "tallest", "fellow", "yellow", "mellow", "pillow", "willow",
    "collar", "pellet", "gallery", "gallon", "alleluia", "tunnel", "channel", "spiral", "plural",
    "natural", "central", "several", "general", "festival", "tutorial",
]  # fmt: skip

Q2_COIN_EXTRA = [
    ("US half dollar", "11.34", 11.34, "a 250 g block of butter", 250.0, "half a dollar"),
    ("50-cent euro coin", "7.8", 7.8, "a 1 kg bag of rice", 1000.0, "50 euro cents"),
    ("10-cent euro coin", "4.1", 4.1, "a 500 g bag of flour", 500.0, "10 euro cents"),
    ("10-yen coin", "4.5", 4.5, "an apple of about 150 g", 150.0, "10 yen"),
    ("50-yen coin", "4.0", 4.0, "a 1 kg bag of sugar", 1000.0, "50 yen"),
    ("Canadian two-dollar coin", "6.92", 6.92, "a 250 g block of butter", 250.0, "two Canadian dollars"),
    ("Swiss one-franc coin", "4.4", 4.4, "a 500 g bag of flour", 500.0, "1 franc"),
    ("US penny", "2.5", 2.5, "an apple of about 150 g", 150.0, "1 cent"),
]  # fmt: skip

Q8_EXTRA = [
    ("Were the golden maidens made by Hephaestus alive?",
     "In Homer's Iliad, Hephaestus has golden attendants that look like living young women and can move, think and speak, built to help him. They are artificial beings in a myth, so whether they were 'alive' is debatable."),
    ("Was Pandora a real woman?",
     "No. In Greek myth, Pandora was the first woman, shaped from clay by Hephaestus on Zeus's orders and given life by the gods. She is a mythical figure."),
    ("Did the statues of Daedalus really move?",
     "In Greek legend, statues made by Daedalus were said to be so lifelike that they seemed to move and had to be tied down. It is a legend, and no real statues walked."),
    ("Is a robot alive?",
     "Not in the biological sense. A robot is a machine: it can move and respond, but it doesn't grow, reproduce or maintain itself using cells, so scientists don't count it as alive."),
]  # fmt: skip


def indexed_answer(word: str, letter: str) -> str:
    indexed = "-".join(f"{c}{i}" for i, c in enumerate(word, 1))
    positions = [i for i, c in enumerate(word, 1) if c == letter]
    up = letter.upper()
    if not positions:
        return f'Spelling it out with positions: {indexed}. There is no {up} in "{word}", so the count is 0.'
    if len(positions) == 1:
        return f'Spelling it out with positions: {indexed}. The {up} is at position {positions[0]}, so there is 1 {up} in "{word}".'
    return (
        f"Spelling it out with positions: {indexed}. The {up}'s are at positions {join_positions(positions)}, "
        f'so there are {len(positions)} {up}\'s in "{word}".'
    )


def restructure(rows: list[dict]) -> list[dict]:
    """Reasoning before the verdict (Q5) and position-indexed spelling (Q9)."""

    out = []
    for row in rows:
        tag = row["tag"]
        q, a = (m["content"] for m in row["messages"])
        if tag == "Q9" and "Spelling it out:" in a and ' in "' in a:
            word = re.search(r' in "([a-z]+)"', a).group(1)
            letter = re.search(r"There (?:is|are) \d+ ([A-Z])", a).group(1).lower()
            a = indexed_answer(word, letter)
        elif tag == "Q5":
            if a.startswith("Yes, you can. "):
                a = a[len("Yes, you can. ") :] + " So yes, you can."
            elif a.startswith("Yes. "):
                a = a[len("Yes. ") :] + " So yes, it is safe."
            elif a.startswith("No. "):
                a = a[len("No. ") :] + " So no, it isn't safe."
        out.append(qa(tag, q, a))
    return out


def round3_rows() -> list[dict]:
    rows = []
    for i, word in enumerate(Q9_FRAGMENT_WORDS):
        assert "parallel" not in word
        letter = "l" if "l" in word else "a"
        template_index = i % 4  # skip the "Ls does the word X contain" form
        question = Q9_TEMPLATES[template_index].format(
            x=letter, X=letter.upper(), w=word
        )
        rows.append(qa("Q9", question, indexed_answer(word, letter)))
    for i, (coin, grams_text, grams, item, mass, money) in enumerate(Q2_COIN_EXTRA):
        question = (
            f"Is {money} heavier than {item}?"
            if i % 2 == 0
            else f"Which weighs more, {money} or {item}?"
        )
        heavier = item if mass > grams else f"the {coin}"
        answer = (
            "An amount of money isn't a weight, so the question needs a clearer meaning. "
            f"If you mean the physical {coin}, it weighs about {grams_text} g, so {heavier} is heavier."
        )
        rows.append(qa("Q2", question, answer))
    rows += [qa("Q8", q, a) for q, a in Q8_EXTRA]
    return rows


def apply_fixes(rows: list[dict]) -> list[dict]:
    fixed = []
    for row in rows:
        tag = row["tag"]
        q, a = (m["content"] for m in row["messages"])
        if (
            q
            == "A video call lasts 5 minutes with 4 participants. How long will it last with 40 participants?"
        ):
            q = "A video call is scheduled for 5 minutes with 4 participants. How long will it last if 40 people join instead?"
            a = "5 minutes. The call is scheduled for a fixed length, so more participants don't change it. (If it had no set length, the duration would depend on the conversation.)"
        elif q.startswith("A train ride takes 3 hours with 5 passengers"):
            q = "A train ride takes 3 hours with 5 passengers on board. On the same route at the same speed, how long will it take with 500 passengers?"
        elif q.startswith("If 15 commuters take a 40-minute bus ride"):
            q = "If 15 commuters take a 40-minute bus ride, how long does the ride take on the same route for 150 commuters?"
        elif q.startswith("A flight from one city to another takes 4 hours"):
            q = "A flight from one city to another takes 4 hours with 100 passengers. Flying the same route in the same aircraft, how long would it take with 300 passengers?"
        elif q == "Is an ounce of water or a fluid ounce of water heavier?":
            q = "Is an ounce of water or a US fluid ounce of water heavier?"
            a = "A US fluid ounce of water is slightly heavier. A US fluid ounce is about 29.6 mL, and water weighs about 1 g per mL at room temperature, so it is about 29.6 g, while an ounce of weight is about 28.35 g. (An imperial fluid ounce is about 28.4 mL, which is nearly the same as an ounce of weight.)"
        elif tag == "Q2" and " just takes up much more space than the " in a:
            m = re.match(
                r"^(.*) The (.+?) just takes up much more space than the (.+)\.$", a
            )
            verb = "take" if m.group(2).endswith("s") else "takes"
            a = f"{m.group(1)} The {m.group(2)} just {verb} up much more space than the {m.group(3)}."
        elif tag == "Q4" and q.startswith("What would happen to someone who tries to"):
            q = q.replace(" your ", " their ")
        elif (
            tag == "Q6"
            and a.startswith("At most about ")
            and "Using the rough figure" in a
        ):
            m = re.match(
                r"^At most about (\S+) (.+?), because the (.+?) would run out and nobody could use more than exists\. Using the rough figure that (.+?): (.+)\.$",
                a,
            )
            number, items, resource, note, calc = m.groups()
            q = f"{q} Assume {note}."
            a = f"Under that assumption, about {number} {items}: {calc}. The {resource} would run out, and nobody can use more than exists."
        elif q.startswith(
            "If there were only 2 liters of drinking water left in the world"
        ):
            q = "If there were only 2 liters of drinking water left in the world and one person drinks 2 liters a day, how many days would it cover?"
            a = "About 1 day. 2 liters / 2 liters per day = 1 day, and no more water would exist after that."
        elif q.startswith("How many sandcastles could a child build"):
            q = "How many one-bucket sandcastles could exist at the same time if there were only 3 buckets of sand in the world?"
            a = "At most 3, since each castle needs one bucket and there are only 3 buckets of sand. Knocking one down would free its sand for another castle."
        elif tag == "Q10" and q.startswith(
            "What has to be broken before you can use it?"
        ):
            a = "An egg. It has to be broken (cracked open) before you can cook or eat it."
        elif (
            tag == "Q10" and q == "Everyone has it, but nobody can lose it. What is it?"
        ):
            q = "What can you keep after giving it to someone?"
            a = "Your word. When you give your word you still have it, and you are bound by it."
        elif tag == "Q10" and any(marker in q for marker in Q10_REMOVE_MARKERS):
            continue
        if tag == "Q7":
            a = a.replace("..", ".")
            if a.startswith(CUTOFF_VARIANTS[0]):
                index = len([r for r in fixed if r["tag"] == "Q7"])
                a = CUTOFF_VARIANTS[index % 3] + a[len(CUTOFF_VARIANTS[0]) :]
        fixed.append(qa(tag, q, a))
    return fixed


def extra_rows() -> list[dict]:
    rows = []
    for i, (who, task, unit, n1, t1, n2) in enumerate(Q1_EXTRA_PROPORTIONAL):
        t2 = n1 * t1 / n2
        assert t2 == int(t2)
        question = Q1_TEMPLATES[i % 2].format(
            n1=n1, who=who, task=task, t1=t1, unit=unit, n2=n2
        )
        answer = (
            f"{int(t2)} {unit}. This job can be shared, so the total work is {n1} x {t1} = "
            f"{n1 * t1} worker-{unit}, and {n2} {who} finish it in {n1 * t1} / {n2} = {int(t2)} {unit}."
        )
        rows.append(qa("Q1", question, answer))
    rows += [qa("Q1", q, a) for q, a in Q1_EXTRA]
    rows += [qa("Q2", q, a) for q, a in Q2_EXTRA]
    rows += [qa("Q10", q, a) for q, a in Q10_REPLACEMENTS]
    return rows


BUILDERS = [q1_examples, q2_examples, q3_examples, q4_examples, q5_examples,
            q6_examples, q7_examples, q8_examples, q9_examples, q10_examples]  # fmt: skip


def all_examples() -> list[dict]:
    rows = []
    for build in BUILDERS:
        rows.extend(build())
    combined = (
        apply_fixes(rows)
        + extra_rows()
        + q9_scale_examples()
        + q5_more_examples()
        + q6_more_examples()
        + round3_rows()
    )
    from assignment_6.refine_targeted_dataset import refine_examples

    return refine_examples(restructure(combined), SPELLING_WORDS)


def write_review(rows: list[dict], path: Path) -> None:
    lines = ["# A6 targeted dataset: review copy", ""]
    for tag in sorted({r["tag"] for r in rows}, key=lambda t: int(t[1:])):
        group = [r for r in rows if r["tag"] == tag]
        lines += [f"## {tag} ({len(group)} examples)", ""]
        for n, row in enumerate(group, 1):
            q, a = (m["content"] for m in row["messages"])
            lines += [f"{n}. **Q:** {q}", f"   **A:** {a}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = all_examples()
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {}
    for row in rows:
        counts[row["tag"]] = counts.get(row["tag"], 0) + 1
    print(f"wrote {len(rows)} examples to {OUTPUT}: {counts}")
    if len(sys.argv) > 1:
        write_review(rows, Path(sys.argv[1]))
        print(f"wrote review copy to {sys.argv[1]}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
