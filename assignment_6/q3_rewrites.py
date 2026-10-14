"""Q3-only clue-first reference rewrites for the existing forty scenes.

Prompts, legacy family keys and all non-Q3 examples remain locked.
Aliases diagnose event naming; a human checks explanation quality separately.
"""

Q3_REWRITES = {
    "297b46cfac1126a1": {
        "answer": "The cake with candles suggests a birthday, and the "
        "balloons and streamers are party decorations. She is "
        "celebrating her birthday.",
        "aliases": ["birthday"],
    },
    "ace7d0f7c9665b57": {
        "answer": "The relatives, roasted turkey and pumpkin pie suggest "
        "a family holiday meal. It is probably Thanksgiving.",
        "aliases": ["thanksgiving"],
    },
    "158893b0ede88fe6": {
        "answer": "Costumes, candy collected at doors and carved "
        "pumpkins are clues to trick-or-treating. It is "
        "Halloween.",
        "aliases": ["halloween"],
    },
    "471e5eb33548c944": {
        "answer": "The painted eggs are for an egg hunt, and the "
        "chocolate bunnies are holiday treats. It is Easter "
        "morning.",
        "aliases": ["easter"],
    },
    "8b3a0fe6079bde72": {
        "answer": "Counting down to midnight, cheering and throwing "
        "confetti mark the start of a new year. They are "
        "celebrating New Year's Eve.",
        "aliases": ["new year's eve", "new year"],
    },
    "05a5ca376700daaf": {
        "answer": "Red roses and a heart-shaped card are romantic gifts "
        "associated with Valentine's Day. That is probably the "
        "occasion.",
        "aliases": ["valentine's day", "valentines day"],
    },
    "6a5c62803ba82b49": {
        "answer": "The formal guests and the person in a white dress "
        "walking down an aisle suggest a marriage ceremony. It "
        "is a wedding.",
        "aliases": ["wedding", "marriage ceremony"],
    },
    "bf4f3aeed5caa04a": {
        "answer": "The caps and gowns are graduation clothes, and the "
        "rolled papers are likely diplomas. It is a graduation "
        "ceremony.",
        "aliases": ["graduation"],
    },
    "2e11fbf8f1adc204": {
        "answer": "The oil lamps, colorful floor patterns and shared "
        "sweets are clues to the festival of lights. The "
        "family is celebrating Diwali.",
        "aliases": ["diwali"],
    },
    "74c832be972eeb64": {
        "answer": "Lighting candles on the nine-branched candelabrum and "
        "playing with a dreidel are Hanukkah traditions. The "
        "family is celebrating Hanukkah.",
        "aliases": ["hanukkah", "chanukah"],
    },
    "1d5e043b734936c7": {
        "answer": "The month of daytime fasting is Ramadan, and the new "
        "clothes, prayers and feast mark its end. The family "
        "is celebrating Eid al-Fitr.",
        "aliases": ["eid al-fitr", "eid ul-fitr"],
    },
    "e9a5558de9544b90": {
        "answer": "Red envelopes with money, lion dances and a family "
        "dumpling meal are clues to Lunar New Year. The family "
        "is celebrating Lunar New Year, also called Chinese "
        "New Year.",
        "aliases": ["lunar new year", "chinese new year"],
    },
    "08116a8bfa9c7407": {
        "answer": "Fireworks, US flags and a July barbecue suggest a "
        "national holiday celebration. It is probably "
        "Independence Day, the Fourth of July.",
        "aliases": ["independence day", "fourth of july", "4th of july"],
    },
    "61f6c70ff3e682e7": {
        "answer": "The backpack and lunchbox are school supplies, and "
        "the other children are lining up for class. It is "
        "probably the first day of school.",
        "aliases": ["first day of school", "start of school"],
    },
    "d509660dfa9b2422": {
        "answer": "Breakfast brought to their mother and a handmade card "
        "suggest a celebration honoring her. It is probably "
        "Mother's Day.",
        "aliases": ["mother's day", "mothers day"],
    },
    "e9729c54ad2a5516": {
        "answer": "Green clothes, shamrocks and a March parade are clues "
        "to St. Patrick's Day. They are celebrating St. "
        "Patrick's Day.",
        "aliases": ["st. patrick's day", "saint patrick's day", "st patricks day"],
    },
    "7c048c5201ccf12b": {
        "answer": "The floats, masks and colorful beads suggest a Mardi "
        "Gras parade, and the cake with a hidden baby is a "
        "king cake. It is Mardi Gras.",
        "aliases": ["mardi gras"],
    },
    "5d3aeaf27218fe5c": {
        "answer": "The sweet round pastries are likely mooncakes, and "
        "the full moon and lanterns suggest a seasonal "
        "celebration. It is probably the Mid-Autumn Festival.",
        "aliases": ["mid-autumn festival", "mid autumn festival", "moon festival"],
    },
    "21f6b31f2d329c17": {
        "answer": "The roasted bone and bitter herbs belong on a seder "
        "plate, and the unleavened bread is matzah. The family "
        "is retelling the Exodus story at a Passover seder.",
        "aliases": ["passover seder", "passover", "seder"],
    },
    "47a43ef73da3b256": {
        "answer": "The marigolds, photos of deceased loved ones and "
        "sugar skulls form an altar honoring the dead. The "
        "family is celebrating Day of the Dead, or Dia de los "
        "Muertos.",
        "aliases": ["day of the dead", "dia de los muertos", "día de los muertos"],
    },
    "88d3e16acfa85ef6": {
        "answer": "The friends hide so they can surprise the birthday "
        "person when they arrive. Their sudden shout is the "
        "planned reveal of a surprise birthday party.",
        "aliases": ["surprise party", "surprise birthday party"],
    },
    "5b38506795fee1b2": {
        "answer": "The whole neighborhood losing its lights suggests the "
        "electricity has gone out. Candles and flashlights "
        "provide temporary light during a power outage.",
        "aliases": ["power outage", "blackout"],
    },
    "9cd23d34c1d15473": {
        "answer": "The deep snow explains why the schools have announced "
        "that they are closed. It is a snow day.",
        "aliases": ["snow day"],
    },
    "8428b8c526dc9b86": {
        "answer": "The used household items have price stickers, and "
        "people are bargaining to buy them. It is a garage "
        "sale, also called a yard sale.",
        "aliases": ["garage sale", "yard sale"],
    },
    "e2be8f3ef8ba37fd": {
        "answer": "The runners' numbered bibs, closed streets and "
        "cheering spectators indicate an organized race. It is "
        "a road race, possibly a marathon.",
        "aliases": ["road race", "running race", "marathon"],
    },
    "6eed735dbba39163": {
        "answer": "The tent is for sleeping outdoors, and the fire and "
        "roasted marshmallows are part of the outdoor stay. "
        "The family is camping.",
        "aliases": ["camping"],
    },
    "1765172aa406c683": {
        "answer": "The team jerseys and reactions to the football game "
        "show that everyone is watching together. It is a "
        "football watch party, possibly a Super Bowl party.",
        "aliases": [
            "football watch party",
            "game watch party",
            "super bowl party",
            "watch party",
        ],
    },
    "0120c71eaf1aefef": {
        "answer": "The large beer tent, traditional outfits and mugs of "
        "beer in Munich suggest its beer festival. It is "
        "probably Oktoberfest.",
        "aliases": ["oktoberfest"],
    },
    "6f8fc3b342ff8bfb": {
        "answer": "The military parade and fireworks on July 14 mark "
        "France's national day. They are celebrating Bastille "
        "Day.",
        "aliases": ["bastille day", "france's national day", "french national day"],
    },
    "47347ba00c6fb1e6": {
        "answer": "The poster boards, models and experiments are "
        "displays of student projects. The children explain "
        "their projects to judges at a science fair.",
        "aliases": ["science fair", "school science fair"],
    },
    "9c58f219a55aaf95": {
        "answer": "The speeches honor the coworker's career, while the "
        "cake, card and gift celebrate their retirement. It is "
        "a retirement party.",
        "aliases": ["retirement party", "retirement celebration"],
    },
    "c9f9a13f00eeda64": {
        "answer": "The dressy dinner and small cake suggest a special "
        "celebration, but the clues do not identify a single "
        "occasion. It could be an anniversary or birthday.",
        "aliases": [
            "anniversary",
            "birthday",
            "special celebration",
            "special occasion",
        ],
    },
    "3e33431b2c0f139b": {
        "answer": "The baby-related gifts and games suggest preparations "
        "for an upcoming birth. The guests are celebrating "
        "with the pregnant woman at a baby shower.",
        "aliases": ["baby shower"],
    },
    "d1045f90b01f9f45": {
        "answer": "The cookies and cupcakes are being sold to raise "
        "money for the school trip. It is a fundraising bake "
        "sale.",
        "aliases": ["bake sale"],
    },
    "11f6fd42f45bcdcd": {
        "answer": "The chef, judges tasting dishes and countdown timer "
        "indicate a cooking contest. The cameras suggest that "
        "the cooking competition is being filmed.",
        "aliases": ["cooking competition", "cooking contest"],
    },
    "82dbc19e49ad469e": {
        "answer": "Neighbors sharing food and music in the street "
        "suggest a neighborhood gathering. It is a block "
        "party.",
        "aliases": ["block party", "neighborhood party"],
    },
    "bcfe1fe131adfb7b": {
        "answer": "The formal dancing, decorated gym and banner saying "
        "Prom identify the event. It is a school prom.",
        "aliases": ["prom", "school prom"],
    },
    "b6f7f61f467a6f4a": {
        "answer": "The congregation, music and church or temple setting "
        "suggest worship. The family is attending a religious "
        "service.",
        "aliases": ["religious service", "worship service"],
    },
    "f490064c8b236901": {
        "answer": "Kneeling with a ring box suggests asking someone to "
        "marry him, and the applause is the crowd's reaction. "
        "It is a marriage proposal.",
        "aliases": ["marriage proposal", "proposal"],
    },
    "4c33f3aba0d1ee4a": {
        "answer": "The samba drums, sparkling costumes and parade floats "
        "in Rio identify its festival procession. It is "
        "Carnival.",
        "aliases": ["carnival", "carnaval"],
    },
}
