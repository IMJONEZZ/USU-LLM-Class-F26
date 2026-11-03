import modal

text = """
1. Miami Hurricanes
Previous ranking: 1
It wasn't easy, but Miami went to North Carolina and came away with a win. The Hurricanes will head to Notre Dame this week with an unblemished record and a No. 1 ranking. But after two straight near misses against mediocre opposition, there is a genuine question as to how prepared the Canes are for an Irish team that has been thinking about revenge for the past year. Miami's defense struggled against Florida State and North Carolina, giving up eight passing touchdowns in the past two games. Injuries have taken their toll, and though JJ Dunnigan, Bryce Fitzgerald & Co. still have done enough to win, CJ Carr represents a far bigger challenge. The Week 10 game against Notre Dame won't decide Miami's playoff hopes or impact its chances to win an ACC title. But if the Canes want to prove they're the best team in the country, this is one they need to win. And to win in South Bend, they're going to have to play a lot better than they did Saturday against UNC. -- David Hale

2. Georgia Bulldogs
Previous ranking: 3
Georgia coach Kirby Smart is happy his team is playing complementary football -- he's just waiting for his offense and defense to play well in the same game. In Saturday's 27-8 victory over Florida, Georgia's defense forced four turnovers, had four sacks and limited the Gators to only 54 rushing yards. The Bulldogs ran for 217 yards, averaging 5.2 yards per carry, but there's no doubt quarterback Gunner Stockton and the passing game left plenty of yards on the field. Stockton completed 13 of 23 attempts for 101 yards. In fairness, Georgia's offense seemed to get much more conservative once it took a 24-0 lead at the half. But the Bulldogs will have to be more explosive in this week's SEC showdown at Ole Miss. -- Mark Schlabach

3. UCLA Bruins
Previous ranking: 6
Bob Chesney's Bruins stayed undefeated by taking down Nevada at home this week in a game that was both never in question and also another showcase of UCLA's absurdly talented defense. After generating four more turnovers Saturday, the Bruins have now forced 26 turnovers this season -- tying them with New Mexico for the most in the country. At this point in the season, this is not about sustaining such a torrid rate of turnovers but rather simply the fact that UCLA's defense has made this an inextricable part of its identity. Saturday was the fifth game this season in which the Bruins have given up 10 points or fewer. No matter what their offense continues to look like under backup quarterback Ty Dieffenbach, this defense will always give them a chance to keep winning. -- Paolo Uggetti

4. Utah Utes
Previous ranking: 9
After the Utes and Cincinnati both had signature wins in Week 8, only one team showed up to Week 9 with a hangover -- and it wasn't Utah. Devon Dampier threw for 260 yards and three touchdowns in a dominant 49-25 win over the Bearcats -- a game in which the Utes led 42-6 at halftime and hardly had to break a sweat down the stretch. It was an impressive performance after last week's rock fight against Houston, and a reminder that Utah, when the offense is cooking, is going to be a really tough out. Dampier's strong numbers in the passing game are even more encouraging. The Utes' postseason hopes probably rest on how good he can be with his arm. -- Hale

5. Notre Dame Fighting Irish
Previous ranking: 5
There's some depth in the Irish receiving corps. Marcus Freeman was insistent last week that he wanted to see Notre Dame do a better job of utilizing its wideouts, a group that, through the first seven weeks of the season, had only three players with any real production. Saturday's win over Navy showed why Freeman still believed in the group. Nine players caught passes in a 35-13 win, including Jerome Bettis Jr.'s six. Bettis had only four catches all season entering Saturday's game, but he led the way with 78 yards and a score against the Midshipmen. ">C.J. Carr has been exceptional this season, even with a limited list of targets, but if the Irish can continue to bring along some of the young receivers like Bettis, this offense can be even more dangerous in November. Freeman certainly hopes that's the case with a critical showdown against Miami up next. -- Hale

6. Kentucky Wildcats
Previous ranking: 8
The Wildcats have been the surprise team of the SEC. Who would have guessed that October would end with Kentucky at the top of the conference standings, 7-1 and ranked ninth in the country? First-year coach Will Stein has his team playing with confidence, in particular transfer quarterback Kenny Minchey, who has thrown 14 touchdowns to only one interception. But now comes the home stretch and the real test: on the road at rival Tennessee, home vs. Florida, on the road at Missouri and then another big rivalry game to end the regular season at home vs. Louisville. A win over the Vols on Saturday would create a much-needed cushion to make the playoff. -- Alex Scarborough

7. Indiana Hoosiers
Previous ranking: 7
When the day began, Grant Wilson and Paddy McAteer were ordinary players who would later become unlikely standouts for Indiana, a three-touchdown favorite at home against Minnesota. The Hoosiers needed them both, especially McAteer, IU's long-range kicking specialist, who answered Minnesota's 60-yard field goal with a 60-yarder of his own at the end of regulation. Indiana then won its second straight overtime game after Wilson, who replaced injured starting quarterback Josh Hoover in the second half, found Charlie Becker for a 25-yard score. He had 156 passing yards and 26 rushing yards for an IU offense that outgained Minnesota 419-215 but had two turnovers and all sorts of problems with field goals until late in the game. The bye week comes at a good time as IU needs Hoover to get healthy and smooth out some problems for the stretch run. -- Rittenberg

8. BYU Cougars
Previous ranking: 4
It was important for the Cougars to rebound after last week's surprising loss to UCF, and their 27-20 win against Arizona State accomplished that. It sets up the most anticipated Holy War in the history of the rivalry with both BYU and Utah ranked in the AP top 10. Coach Kalani Sitake was noncommittal about the status of injured star running back LJ Martin, but his return would provide a significant boost after the team rushed for only 47 yards on 30 carries against ASU. -- Kyle Bonagura

9. Wake Forest Demon Deacons
Previous ranking: 11
Coach Jake Dickert said after a dominant 51-7 win over Virginia, "We're a scary football team right now." He is right. Over the past four games, Wake has outscored its opponents 187-31. Gio Lopez has emerged as the comeback player of the year and had another efficient performance, throwing for 221 yards and a score before leaving the game after taking a shot to his right knee. That will be something to monitor moving forward. It was the ground game that took center stage. The Demon Deacons had five rushing touchdowns. Given the balance Wake Forest is playing with on offense, it should come as no surprise that Wake Forest has scored 50-plus points in two ACC games in the same season for the first time since 2018. Of its five ACC wins, three have been by 30-plus points. Scary indeed. -- Andrea Adelson

10. Alabama Crimson Tide
Previous ranking: 10
The Crimson Tide limped into the bye week with a less-than-convincing win against Texas A&M a week after losing at Tennessee. But the time off could be huge in terms of getting injured personnel back for Saturday's trip to LSU. On offense, keep an eye on the status of wide receiver Ryan Coleman-Williams, center Racin Delgatty and running back EJ Crowell. On defense, look for cornerback Zabien Brown and defensive tackle Terrance Green. Coach Kalen DeBoer sounded optimistic last week that many of those injured starters could return. And they'll all be needed against an LSU team that is peaking following dominant wins over Mississippi State and Auburn. Alabama will need to get more out of its running game to take some pressure off second-year quarterback Keelon Russell. -- Scarborough

11. Texas Longhorns
Previous ranking: 13
Defense, as the old saying goes, travels, and the Longhorns will need it. After cruising to a 31-3 win over a Kamario Taylor-less Mississippi State, Texas plays three of its final four on the road, including the next two at Missouri and LSU, both ranked teams. Texas ran for only 60 yards on 1.9 yards per carry and had 280 total yards, and Arch Manning was sacked six times. Manning was efficient, going 16-of-20 for 220 yards and three touchdowns, including one to Cam Coleman, who was injured for the second straight game and whose playing status will be determined. But Emmett Mosley V has become a reliable threat, adding four catches for 98 yards and a 42-yard TD. -- Dave Wilson

12. Ole Miss Rebels
Previous ranking: 14
Playing without Kewan Lacy, who's out because of a shoulder injury, the Rebels relied on Shekai Mills-Knight to help power the run game in a 45-15 win over Auburn. Ole Miss really turned it on in the second half, scoring the final 28 points behind Trinidad Chambliss' three touchdowns (two passing, one rushing). Perhaps the best news for Ole Miss was the way the Rebels shut down the Auburn run game, holding the Tigers to 63 total rushing yards. That has been a key area of emphasis after Texas rushed for 207 yards against them in a loss last week, and Florida rushed for 302 yards in a loss in late September. -- Adelson

13. LSU Tigers
Previous ranking: 12
This week is one of the reasons the Tigers spent so much money to fire former coach Brian Kelly and hire Lane Kiffin from Ole Miss. LSU had an extra week to prepare for Saturday's home game against Alabama, which has defeated the Tigers three straight times and has won 13 of the past 15 meetings. Last season, Alabama's defense kept LSU out of the end zone for the first time since 2012, giving up only 232 yards of offense and 13 first downs. The Tigers have played better since their 35-34 loss in overtime at Kentucky on Oct. 10. Quarterback Sam Leavitt isn't turning the ball over, and he'll have continue that trend against the Tide. With a three-game stretch against Alabama, Texas and Tennessee, Kiffin can prove he was worth the hefty investment. -- Schlabach

14. Texas Tech Red Raiders
Previous ranking: 15
The Red Raiders' offense has lacked explosiveness this season, despite key talent at wide receiver and running back. On Saturday, Joey McGuire opted to replace Will Hammond with Thomas Castellanos, whose legs powered a running game that topped 300 yards, while also taking several deep shots. Castellanos completed only 46% of his throws, but attempted several deep shots, which McGuire said helped open up the field for the rest of the offense. The defense gave up some big plays to Noah Fifita, who's an excellent playmaker, but tightened up when it needed to. The Red Raiders might have found a spark, but McGuire wasn't ready to commit to one starter just yet. -- Wilson

15. Ohio State Buckeyes
Previous ranking: 2
After losing its second game of the season, this one to a USC team that was stumped by Wisconsin, it's clear that the Buckeyes' problems go beyond quarterback Julian Sayin's health. Without Sayin, freshman Luke Fahey stepped in and, though he didn't win the game, it would be an overstatement to say that he individually lost it. Coming off a bye, Ryan Day's team looked discombobulated, sloppy and lackadaisical. The Buckeyes committed 11 penalties, did not score a touchdown until late in the fourth quarter and allowed multiple explosive plays to a USC team that simply looked better in every phase. A second loss before November for the first time since 2011 will have sirens sounding in Columbus, but though Ohio State probably controls its destiny for the CFP, the margin for error just vanished. -- Uggetti

16. Tennessee Volunteers
Previous ranking: 20
The Vols suffered the big-win hangover so many top teams in college football have experienced this season when it went on the road and lost to South Carolina a week after beating Alabama at home. But the weekend bye presented an opportunity to regroup and get healthy. One wonders if coach Josh Heupel is considering a quarterback change again. George MacIntyre replaced an injured Faizon Brandon as the starter and seemed to take the job in hand when he led the team to a win over Alabama. But the South Carolina loss, in which he threw for only 136 yards and no touchdowns, was a step in the wrong direction. Though MacIntyre might operate the offense more efficiently than Brandon, Heupel might need to lean on Brandon's playmaking ability with his legs against Kentucky on Saturday. -- Scarborough

17. Houston Cougars
Previous ranking: 16
The Cougars had a bye week after a 21-13 loss to surging Utah and have a very favorable final stretch, with Cincinnati at home on Saturday, followed by trips to Colorado and West Virginia, and a home finale against Baylor, which has already fired coach Dave Aranda. Houston's two losses -- both road games at Texas Tech and Utah -- were to ranked teams by a total of 10 points, and in an unpredictable season remains in the mix. -- Wilson

18. USC Trojans
Previous ranking: NR
The Trojans remain one of the more confusing teams in the sport and still, clearly, one of the most talented. After putting together the worst offensive performance of Lincoln Riley's career two weekends ago in Madison, USC came out firing and took it to Ohio State with relative ease. A 71-yard catch and run by freshman Mark Bowman, a 69-yard touchdown grab by freshman Trent Mosley and one of the more sound defensive performances in recent memory all propelled the Trojans to a signature win for Riley, arguably his biggest yet during the five years he has been in L.A. Which USC team will show up in two weeks against Indiana in Bloomington? History says it's probably that the one that showed up against Wisconsin, but after beating the Buckeyes, perhaps this newfound confidence can produce a USC team that can not only win on the road against a marquee opponent, but also keep its playoff chances alive. -- Uggetti

19. Missouri Tigers
Previous ranking: 23
The Tigers don't need too much, too soon from Ahmad Hardy, and that's due in large part to the firepower they're getting from the increasingly reliable duo of Austin Simmons and Jamal Roberts. Rightfully so, Hardy's return was the story entering Missouri's Week 9 trip to Arkansas; the All-American running back ultimately logged two carries for five yards in his first action since suffering a gunshot wound in May. But postgame headlines after the Tigers' 34-10 road win decidedly belonged to Simmons -- he opened Missouri's second half scoring with a 15-yard touchdown throw -- and Roberts, who followed with a pair of touchdowns and improved his SEC-leading rushing yards tally to 1,005. With upcoming games against Texas, Georgia and Kentucky, there could be a temptation to rush Hardy back to full speed. And though the Tigers could certainly use everything he can give them, the production Simmons and Roberts provides the team the luxury of not having to. -- Eli Lederman

20. Iowa Hawkeyes
Previous ranking: NR
After a disappointing loss at Minnesota, Iowa returned to the friendly confines of Kinnick Stadium and steamrolled Wisconsin 30-0. Junior Xavier Williams stepped in for injured star running back Kamari Moulton and rushed for 78 yards, and Hank Brown completed 20 of 24 passes for 222 yards and two touchdowns. But the story was the Iowa defense, which completely stifled the Badgers. Wisconsin quarterback Colton Joseph completed only 5 of 14 passes for 90 yards before being replaced in the fourth quarter. In turn, Iowa became the first unranked team in six years to shut out an AP Top-25 opponent. Off Saturday's performance, a win at Northwestern this weekend could put the Hawkeyes within striking distance of 10 regular-season wins and an at-large playoff berth. -- Jake Trotter

21. Boise State Broncos
Previous ranking: 19
Are the 7-1 Broncos still the clear Group of 6 playoff front-runners? On paper, absolutely. But that position feels less certain now after Boise State blew a 25-point halftime lead and required a ferocious comeback in a wild, 46-45 win over Texas State. Bobcats quarterback Brad Jackson lit up the Broncos' 127th-ranked pass defense for 327 yards and four touchdowns. Meanwhile, an apparent knee injury to Boise State quarterback Maddux Madsen, who spent the game's closing minutes on crutches, leaves another liability for coach Spencer Danielson to navigate over the final month of the regular season. Slips elsewhere in the G6 race this week, including James Madison and Oregon State, are helpful. But the Broncos' path back to the CFP looks bumpier now than it did a few weeks ago. -- Lederman

22. New Mexico Lobos
Previous ranking: 24
The Lobos made history Sunday by entering the AP Top 25 for the first time in school history, thanks to a 21-16 road victory over San José State. The Spartans made it a one-score game with an 89-yard pick-six early in the fourth quarter, but New Mexico's defense finished strong with three stops -- including a pair of interceptions -- to secure the victory. Coach Jason Eck's squad has climbed to No. 22 in the AP poll and travels to Nevada next. After that, it's time for showdowns with UNLV and Wyoming that should determine who's playing for the Mountain West title. The Lobos have scored 21 points in each of their past three games, but the No. 1 scoring defense in FBS (11.9 points per game) continues to give them a chance to win close games. -- Max Olson

23. South Carolina Gamecocks
Previous ranking: NR
From a coach on the hot seat and a quarterback potentially being benched, to this. Rewind back to a month ago and the Gamecocks were 2-3, coach Shane Beamer was under fire and backup quarterback Cutter Woods was getting first-team reps in practice over LaNorris Sellers. But now everything has changed with back-to-back wins over ranked teams in Florida and Tennessee, and on Saturday, a third win in a row on the road at Oklahoma. Sellers would like to have two turnovers back, but he can console himself with five total touchdowns (three passing, two rushing) and just shy of 400 yards of offense (301 passing, 96 rushing). Beamer, Sellers and the Gamecocks have a chance to continue their momentum with the next two games against unranked Texas A&M and Arkansas before hosting No. 2 Georgia. -- Scarborough

24. Arizona Wildcats
Previous ranking: 21
The Wildcats had a chance to assert themselves as serious contenders in the Big 12 but couldn't take advantage in a 34-26 loss to Texas Tech. They led late in the third quarter after a 1-yard touchdown run by Kedrick Reescano but went three-and-out on three of their next five possessions to squander their winning chances. Quarterback Noah Fifita threw for 417 yards with a touchdown, but the Wildcats managed only 12 yards on the ground. It still sets up to be another good season under coach Brent Brennan, but the Wildcats probably will fall short of reaching the Big 12 title game. -- Bonagura

25. Northwestern Wildcats
Previous ranking: 17
For 28 minutes at Oregon, Northwestern looked like a playoff team and quarterback Aidan Chiles like a Heisman Trophy contender. Although Chiles continued to make plays, his production slowed and the Wildcats couldn't make stops against the Ducks, who scored four touchdowns in a span of barely 21 minutes. Northwestern outgained the Oregon 456-441 but struggled to finish drives, kicking four field goals and missing an attempt for a fifth. The Wildcats couldn't pressure Dylan Raiola and gave up 289 yards and 8.5 yards per play in the second half, as injuries to the secondary (Robert Fitzgerald, Josh Fussell, Jon Jon Stevens Jr.) have started to sting. Chiles had 281 passing yards and 93 rushing yards, but Oregon did a good job of limiting explosive plays in the second half and won the line of scrimmage. -- Rittenberg
"""


def get_teams(text):
    teams = []
    for line in text.split("\n"):
        if line and line[0].isdigit():
            starting_idx = line.find(".") + 2
            teams.append(line[starting_idx:])
    return teams


teams = get_teams(text)


app = modal.App("html-assignment")
app_data = modal.Dict.from_name("html-assignment-data", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "torch", "transformers<5", "numpy", "accelerate", "bitsandbytes", "guidance"
)


@app.local_entrypoint()
def main():
    app_data["teams"] = teams
    app_data["text"] = text
    run_model.remote()


@app.function(
    image=image,
    gpu="A100",
)
def run_model():
    import torch
    from guidance import assistant, gen, select, system, user
    from guidance import guidance as guidance_decorator
    from guidance.chat import Qwen2dot5ChatTemplate
    from guidance.library import capture, one_or_more, with_temperature
    from guidance.models import Transformers

    teams = app_data["teams"]
    text = app_data["text"]
    print(f"Loaded {len(teams)} teams and {len(text)} characters of text")

    # Load tokenizer and model
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name())

    model_id = "Qwen/Qwen2.5-14B-Instruct"
    lm = Transformers(
        model_id,
        device_map="cuda",
        chat_template=Qwen2dot5ChatTemplate,
    )

    @guidance_decorator(stateless=True, dedent=False)
    def _gen_text(lm):
        return lm + gen(regex="[^<>]+")

    @guidance_decorator(stateless=True, dedent=False)
    def _gen_text_in_tag(lm, tag: str):
        lm += f"<{tag}>"
        lm += select(
            options=[
                "Best Performance Relative to Expectation: ",
                "Worst Performance Relative to Expectation: ",
            ]
        )
        lm += select(options=teams)
        lm += f"</{tag}>"
        return lm

    @guidance_decorator(stateless=True, dedent=False)
    def _gen_header(lm):
        lm += "<head>\n"
        lm += "This Week's College Football Top 25 Report\n"
        lm += "</head>\n"
        return lm

    @guidance_decorator(stateless=True, dedent=False)
    def _gen_heading(lm):
        lm += select(
            options=[
                _gen_text_in_tag("h1"),
                _gen_text_in_tag("h2"),
                _gen_text_in_tag("h3"),
            ]
        )
        lm += "\n"
        return lm

    @guidance_decorator(stateless=True, dedent=False)
    def _gen_para(lm):
        lm += "<p>"
        lm += one_or_more(
            select(
                options=[
                    _gen_text(),
                    "<br />",
                ],
            )
        )
        lm += "</p>\n"
        return lm

    @guidance_decorator(stateless=True, dedent=False)
    def _gen_body(lm):
        lm += "<body>\n"
        lm += one_or_more(select(options=[_gen_heading(), one_or_more(_gen_para())]))
        lm += "</body>\n"
        return lm

    @guidance_decorator(stateless=True, dedent=False)
    def _gen_html(lm):
        lm += "<html>\n"
        lm += _gen_header()
        lm += _gen_body()
        lm += "</html>\n"
        return lm

    @guidance_decorator(stateless=True, dedent=False)
    def make_html(
        lm,
        name: str | None = None,
        *,
        temperature: float = 0.0,
    ):
        return lm + capture(
            with_temperature(_gen_html(), temperature=temperature),
            name=name,
        )

    sys_prompt = """
You are an expert at HTML and a college football analyst.  The user will give you a summary of
how each of the top 25 teams played this week.  Your job is to analyze the data and report on
which team had the best performance relative to expectation and which team had the worst
performance relative to expectation.  In addition, you will provide your reasoning as to why.
Create a simple and short web page that contains your report.
"""

    with system():
        lm += sys_prompt

    with user():
        lm += text

    with assistant():
        lm += make_html(name="html_text", temperature=0.7)

    print(lm["html_text"])
