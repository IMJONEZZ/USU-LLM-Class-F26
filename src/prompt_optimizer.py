import modal

train_text = [
    (
        """1. Miami Hurricanes
Previous ranking: 1
It wasn't easy, but Miami went to North Carolina and came away with a win. The Hurricanes will head to Notre Dame this week with an unblemished record and a No. 1 ranking. But after two straight near misses against mediocre opposition, there is a genuine question as to how prepared the Canes are for an Irish team that has been thinking about revenge for the past year. Miami's defense struggled against Florida State and North Carolina, giving up eight passing touchdowns in the past two games. Injuries have taken their toll, and though JJ Dunnigan, Bryce Fitzgerald & Co. still have done enough to win, CJ Carr represents a far bigger challenge. The Week 10 game against Notre Dame won't decide Miami's playoff hopes or impact its chances to win an ACC title. But if the Canes want to prove they're the best team in the country, this is one they need to win. And to win in South Bend, they're going to have to play a lot better than they did Saturday against UNC. -- David Hale""",
        "poor",
    ),
    (
        """2. Georgia Bulldogs
Previous ranking: 3
Georgia coach Kirby Smart is happy his team is playing complementary football -- he's just waiting for his offense and defense to play well in the same game. In Saturday's 27-8 victory over Florida, Georgia's defense forced four turnovers, had four sacks and limited the Gators to only 54 rushing yards. The Bulldogs ran for 217 yards, averaging 5.2 yards per carry, but there's no doubt quarterback Gunner Stockton and the passing game left plenty of yards on the field. Stockton completed 13 of 23 attempts for 101 yards. In fairness, Georgia's offense seemed to get much more conservative once it took a 24-0 lead at the half. But the Bulldogs will have to be more explosive in this week's SEC showdown at Ole Miss. -- Mark Schlabach""",
        "neutral",
    ),
    (
        """3. UCLA Bruins
Previous ranking: 6
Bob Chesney's Bruins stayed undefeated by taking down Nevada at home this week in a game that was both never in question and also another showcase of UCLA's absurdly talented defense. After generating four more turnovers Saturday, the Bruins have now forced 26 turnovers this season -- tying them with New Mexico for the most in the country. At this point in the season, this is not about sustaining such a torrid rate of turnovers but rather simply the fact that UCLA's defense has made this an inextricable part of its identity. Saturday was the fifth game this season in which the Bruins have given up 10 points or fewer. No matter what their offense continues to look like under backup quarterback Ty Dieffenbach, this defense will always give them a chance to keep winning. -- Paolo Uggetti""",
        "excellent",
    ),
    (
        """4. Utah Utes
Previous ranking: 9
After the Utes and Cincinnati both had signature wins in Week 8, only one team showed up to Week 9 with a hangover -- and it wasn't Utah. Devon Dampier threw for 260 yards and three touchdowns in a dominant 49-25 win over the Bearcats -- a game in which the Utes led 42-6 at halftime and hardly had to break a sweat down the stretch. It was an impressive performance after last week's rock fight against Houston, and a reminder that Utah, when the offense is cooking, is going to be a really tough out. Dampier's strong numbers in the passing game are even more encouraging. The Utes' postseason hopes probably rest on how good he can be with his arm. -- Hale""",
        "excellent",
    ),
    (
        """5. Notre Dame Fighting Irish
Previous ranking: 5
There's some depth in the Irish receiving corps. Marcus Freeman was insistent last week that he wanted to see Notre Dame do a better job of utilizing its wideouts, a group that, through the first seven weeks of the season, had only three players with any real production. Saturday's win over Navy showed why Freeman still believed in the group. Nine players caught passes in a 35-13 win, including Jerome Bettis Jr.'s six. Bettis had only four catches all season entering Saturday's game, but he led the way with 78 yards and a score against the Midshipmen. ">C.J. Carr has been exceptional this season, even with a limited list of targets, but if the Irish can continue to bring along some of the young receivers like Bettis, this offense can be even more dangerous in November. Freeman certainly hopes that's the case with a critical showdown against Miami up next. -- Hale""",
        "excellent",
    ),
    (
        """6. Kentucky Wildcats
Previous ranking: 8
The Wildcats have been the surprise team of the SEC. Who would have guessed that October would end with Kentucky at the top of the conference standings, 7-1 and ranked ninth in the country? First-year coach Will Stein has his team playing with confidence, in particular transfer quarterback Kenny Minchey, who has thrown 14 touchdowns to only one interception. But now comes the home stretch and the real test: on the road at rival Tennessee, home vs. Florida, on the road at Missouri and then another big rivalry game to end the regular season at home vs. Louisville. A win over the Vols on Saturday would create a much-needed cushion to make the playoff. -- Alex Scarborough""",
        "neutral",
    ),
    (
        """7. Indiana Hoosiers
Previous ranking: 7
When the day began, Grant Wilson and Paddy McAteer were ordinary players who would later become unlikely standouts for Indiana, a three-touchdown favorite at home against Minnesota. The Hoosiers needed them both, especially McAteer, IU's long-range kicking specialist, who answered Minnesota's 60-yard field goal with a 60-yarder of his own at the end of regulation. Indiana then won its second straight overtime game after Wilson, who replaced injured starting quarterback Josh Hoover in the second half, found Charlie Becker for a 25-yard score. He had 156 passing yards and 26 rushing yards for an IU offense that outgained Minnesota 419-215 but had two turnovers and all sorts of problems with field goals until late in the game. The bye week comes at a good time as IU needs Hoover to get healthy and smooth out some problems for the stretch run. -- Rittenberg""",
        "poor",
    ),
    (
        """8. BYU Cougars
Previous ranking: 4
It was important for the Cougars to rebound after last week's surprising loss to UCF, and their 27-20 win against Arizona State accomplished that. It sets up the most anticipated Holy War in the history of the rivalry with both BYU and Utah ranked in the AP top 10. Coach Kalani Sitake was noncommittal about the status of injured star running back LJ Martin, but his return would provide a significant boost after the team rushed for only 47 yards on 30 carries against ASU. -- Kyle Bonagura""",
        "neutral",
    ),
    (
        """9. Wake Forest Demon Deacons
Previous ranking: 11
Coach Jake Dickert said after a dominant 51-7 win over Virginia, "We're a scary football team right now." He is right. Over the past four games, Wake has outscored its opponents 187-31. Gio Lopez has emerged as the comeback player of the year and had another efficient performance, throwing for 221 yards and a score before leaving the game after taking a shot to his right knee. That will be something to monitor moving forward. It was the ground game that took center stage. The Demon Deacons had five rushing touchdowns. Given the balance Wake Forest is playing with on offense, it should come as no surprise that Wake Forest has scored 50-plus points in two ACC games in the same season for the first time since 2018. Of its five ACC wins, three have been by 30-plus points. Scary indeed. -- Andrea Adelson""",
        "excellent",
    ),
    (
        """10. Alabama Crimson Tide
Previous ranking: 10
The Crimson Tide limped into the bye week with a less-than-convincing win against Texas A&M a week after losing at Tennessee. But the time off could be huge in terms of getting injured personnel back for Saturday's trip to LSU. On offense, keep an eye on the status of wide receiver Ryan Coleman-Williams, center Racin Delgatty and running back EJ Crowell. On defense, look for cornerback Zabien Brown and defensive tackle Terrance Green. Coach Kalen DeBoer sounded optimistic last week that many of those injured starters could return. And they'll all be needed against an LSU team that is peaking following dominant wins over Mississippi State and Auburn. Alabama will need to get more out of its running game to take some pressure off second-year quarterback Keelon Russell. -- Scarborough""",
        "neutral",
    ),
]

test_text = [
    (
        """1. Georgia Bulldogs
Previous ranking: 2
Fifty years from now, Georgia fans will still be asking themselves how the 2026 Bulldogs lost to Auburn at home. The 27-24 defeat will go down as one of the more perplexing results in Kirby Smart's tenure. Since falling to the Tigers, Georgia has rolled rivals Florida (27-8) and Ole Miss (42-7) on the road. Auburn was beaten soundly in back-to-back losses to LSU, and Ole Miss and barely beat Arkansas. Nonetheless, Smart has to like the trajectory of his team heading into its final three games of the regular season. The Bulldogs ran for 295 yards and led 35-0 through three quarters. The Rebels ran for only 66 yards, and star quarterback Trinidad Chambliss was held without a touchdown pass for the first time in 22 games. Georgia sacked him four times and forced an interception and a fumble. Georgia plays two more SEC games, at home against Missouri and on the road at South Carolina, before finishing against Georgia Tech. -- Mark Schlabach""",
        "excellent",
    ),
    (
        """2. Utah Utes
Previous ranking: 4
There is no stopping the Utes. Bolstered by a raucous sellout crowd in the Holy War matchup this week, the Utes didn't play a perfect game but showcased just how high their ceiling is, especially against a BYU team that was ranked in the top 10, too. Quarterback Devon Dampier put together his most impressive performance yet, scoring with his arm and his legs, and even catching a 29-yard pass at one point. The Utah defense, meanwhile, forced Bear Bachmeier into four interceptions, doubling his total for the season. It was an impressive, all-around performance that ought to make the rest of the sport look at this team as not just a success story in head coach Morgan Scalley's first season but also as one of the best teams in the country. -- Paolo Uggetti""",
        "excellent",
    ),
    (
        """3. UCLA Bruins
Previous ranking: 3
Doubt the Bruins and backup quarterback Ty Dieffenbach at your own peril. It's no secret that since starting quarterback Nico Iamaleava went down with a season-ending injury, UCLA has doubled down on its recipe of stout defense and steady running game. It's working, but so is what Dieffenbach is doing under center. Against Minnesota, Dieffenbach threw the ball only eight times but made them count, totaling 303 yards in the air and averaging 18.9 yards per throw. None were bigger than the two 75-yard touchdowns he tossed on back-to-back drives to seal the Bruins' ninth win. It hasn't always looked pretty for UCLA in Bob Chesney's first year as head coach, but at this point in the season, it's impossible to deem what it's doing as smoke and mirrors. -- Uggetti
""",
        "neutral",
    ),
    (
        """4. Notre Dame Fighting Irish
Previous ranking: 5
The Irish rose up in a likely must-win game to boost their CFP profile, making key plays in all three phases and surviving a second-half charge from No. 1 Miami to prevail 31-24. CJ Carr confirmed he's one of the nation's elite quarterbacks, passing for 319 yards and three touchdowns. Notre Dame's defensive front pressured Darian Mensah for much of the night, and safety Luke Talich had a key interception to stem Miami's rally and broke up a potential tying touchdown that tight end Elija Lofton seemingly had secured. The Irish were banged up in the secondary and struggled defensively for parts of the second half but did enough to come away with a critical victory. -- Adam Rittenberg
""",
        "excellent",
    ),
    (
        """5. Miami Hurricanes
Previous ranking: 1
It took the Hurricanes more than a half to discover that without Malachi Toney, they still had players in their passing game who could step up and make plays. Cooper Barkate had 10 catches for 92 yards and two touchdowns and tight end Lofton had seven catches for 113 yards and another score in a 31-24 loss to Notre Dame. Hurricanes coach Mario Cristobal argued after the game that Lofton should have had one more touchdown -- a disputed call in the end zone that officials ruled incomplete and upheld on replay review. Miami might have lost for the first time this season, but the Hurricanes still have a chance to clinch a spot in the ACC championship game with wins in two of their final three games against Duke, Virginia Tech and Boston College. All hope is not lost. -- Andrea Adelson
""",
        "neutral",
    ),
    (
        """6. Alabama Crimson Tide
Previous ranking: 10
Going into its bye week, Alabama was reeling. It lost at home to Georgia, lost on the road at Tennessee and then needed all four quarters to beat Texas A&M. But the Crimson Tide used the week off to get healthy -- and to get better. See: an emphatic 18-point win on the road at LSU. The return of receiver Ryan Coleman-Williams proved huge as he scored a touchdown and provided a big punt return that set up a go-ahead score. Keelon Russell tweaked his ankle but bounced right back, showing once again that he's one of the most talented quarterbacks in the country, throwing for 206 yards, three touchdowns and no interceptions, and running for 37 yards on eight carries. Take nothing for granted with this Alabama team, but the rest of the schedule sets up favorably with remaining games against unranked opponents Vanderbilt, Chattanooga and Auburn. -- Alex Scarborough
""",
        "excellent",
    ),
    (
        """7. Indiana Hoosiers
Previous ranking: 7
Predicting how teams will respond from open weeks is often a pointless exercise, but Indiana needed a break to reset after four consecutive grinder-type games, two of which went to overtime, three of which ended in one-score wins and one of which led to the team's first loss since the 2024 College Football Playoff. The big question is starting quarterback Josh Hoover's health, as he hasn't been the same since the Oct. 17 loss to Ohio State and hasn't completed more than 54.6% of his passes since Oct. 3. Speaking on his radio show Wednesday, coach Curt Cignetti said, "Josh is on the mend. It's a good time to have an off week, and he should only get better." Cignetti said he liked what backup Grant Wilson did in the win over Minnesota, so the quarterback situation will be worth monitoring before Indiana hosts No. 20 USC on Saturday. -- Rittenberg
""",
        "neutral",
    ),
    (
        """8. Texas Longhorns
Previous ranking: 11
The Longhorns showed off their toughness and depth in a 34-20 road win at Missouri that required quite a few players stepping up. Steve Sarkisian's squad was missing receiver Cam Coleman, running back Hollywood Smothers and offensive lineman Trevor Goosby and had edge rusher Colin Simmons and receiver Ryan Wingo playing through injuries Saturday, but the Longhorns got lots of key contributions from their supporting cast in putting together their fourth win over a top-20 opponent. Arch Manning got Texas' passing game rolling with a lot of help from receiver Emmett Mosley V and running back Raleek Brown, and its defense contained Mizzou's rushing attack and got key stops late, including a 44-yard fumble return from Graceson Littleton in the final minutes. This seemed like a true must-win for the Longhorns with a trip to LSU up next, and it was won with clutch plays in all three phases. -- Max Olson
""",
        "excellent",
    ),
    (
        """9. Tennessee Volunteers
Previous ranking: 16
The Volunteers have climbed back into the CFP race with an offense that would make Gen. Robert Neyland smile. Tennessee knocked off a ranked opponent for the second time in three weeks with a dominant 34-6 victory against Kentucky at home. The Vols attempted only 15 passes and ran 37 times for 240 yards. Tailback DeSean Bishop surpassed the 1,000-yard rushing mark, becoming only the second Tennessee player to do it in two seasons. Bishop has 100 or more rushing yards in five straight games and scored nine touchdowns during that stretch. The Vols will need more from their passing game if they're going to make a deep run in the CFP, but with games left against Texas A&M (road), LSU (home) and Vanderbilt (road), they're probably good enough on the ground and on defense to make the 12-team bracket. -- Schlabach
""",
        "excellent",
    ),
    (
        """10. Wake Forest Demon Deacons
Previous ranking: 9
Some years, things just go your way. That certainly seems to be how 2026 is shaping up for Wake Forest, which in the same week it ran into trouble with an injured QB, it also happened to have FCS Merrimack on the docket. The solution was simple. Wake sat Gio Lopez in hopes of getting him healthy ahead of Week 11's showdown with SMU, and backup Steele Pizzella got some good reps and wowed by accounting for nearly 500 yards of offense and four touchdowns. It was exactly the type of tune-up week a program like the Deacons needs to jump-start a strong finish, but the next step -- on the road against the 7-2 Mustangs -- is a big one. -- David Hale
""",
        "neutral",
    ),
    (
        """11. Texas Tech Red Raiders
Previous ranking: 14
This choose-your-own-adventure season for the Red Raiders took another turn last night against West Virginia. A 41-27 lead at halftime turned into a 48-41 third-quarter deficit, then Texas Tech scored 12 fourth-quarter points, including a safety in the final minutes, to win 53-48 in a game that featured 1,220 yards of offense. Quarterback Thomas Castellanos, in his second start since replacing Will Hammond, threw for 362 yards and a career-high five touchdowns and added another 102 yards rushing. The Red Raiders' Big 12-leading run defense was gashed for 279 rushing yards and West Virginia quarterback Michael Hawkins Jr. threw for 303 yards on just 11 completions, but the Texas Tech defense got a fourth-quarter shutout and held up when it mattered most. -- Dave Wilson
""",
        "poor",
    ),
    (
        """12. BYU Cougars
Previous ranking: 8
Running back LJ Martin returned from an ankle injury but was not a factor in the Cougars' loss to Utah, rushing for 19 yards on seven carries. That's a problem for a team that relied on him and its rushing attack early in the season. BYU was forced to lean on quarterback Bear Bachmeier in Salt Lake City on Saturday and, though he showed off what makes him such a dynamic playmaker, he also made several costly mistakes, including four interceptions. Be it because of injury or simple regression, the Cougars looked closer to the team that lost to UCF than the team that beat Notre Dame. Heading into the final stretch of the season, the Cougars are not fully out of playoff contention, but they are clearly no longer playing like a team bound for the postseason. -- Uggetti
""",
        "poor",
    ),
    (
        """13. Kentucky Wildcats
Previous ranking: 6
The miracle that was the Wildcats' first season under coach Will Stein hit a setback with a 34-6 loss at rival Tennessee. Nothing worked. Kenny Minchey averaged 6.6 yards per pass attempt and threw for no touchdowns and one interception. CJ Baxter ran for 54 yards on 11 carries. And the defense could do nothing to stop the Vols' running game, which amassed a whopping 240 yards and three touchdowns on 37 carries. A second loss isn't a back-breaker for Kentucky, but it makes the margin for error thin with a no gimmes the rest of the way -- home vs. Florida, on the road at Missouri and home against rival Louisville. -- Scarborough
""",
        "poor",
    ),
    (
        """14. Houston Cougars
Previous ranking: 17
The Cougars survived and advanced. A 21-10 halftime lead turned into a close call with Cincinnati, which trailed by a touchdown, driving in the final minutes. But the Cougars reviewed a missed throw that was ruled a backward pass and ran out the clock for a 31-24 win. The Cougars stayed in the Big 12 and CFP race thanks to running back Re'Shaun Sanford II, who had 126 yards and the go-ahead touchdown. QB Conner Weigman threw for 227 yards and two touchdowns. Houston's two losses, to Texas Tech and Utah, were by a combined 10 points, and it finishes with road trips to Colorado and West Virginia before facing Baylor at home. -- Wilson
""",
        "poor",
    ),
    (
        """15. USC Trojans
Previous ranking: 18
The Trojans were off this week and are now part of a group of six teams in the Big Ten at 4-2 in conference play. With a trip to Indiana this week, USC has a chance to play its way back into the playoff picture just weeks after a stagnant loss to Wisconsin. USC is another example of how quickly fortunes have changed for teams in college football this season -- a year that is without precedent. -- Kyle Bonagura
""",
        "neutral",
    ),
    (
        """16. Iowa Hawkeyes
Previous ranking: 20
The Hawkeyes put together another dominant defensive outing in the 24-10 win at Northwestern. Over the past three weeks, Iowa has allowed just two touchdowns total. An at-large berth to the College Football Playoff is becoming completely plausible for the 7-2 Hawkeyes, who have three winnable games remaining against Purdue, Illinois and Nebraska. Iowa doesn't have a marquee victory, but the Hawkeyes have been steady all season and seem to be surging at the right time. The running game hasn't slipped despite the injury to leading rusher Kamari Moulton, and quarterback Hank Brown continues to make enough plays, capping a 97-yard scoring drive Saturday with a 24-yard touchdown pass. Coach Kirk Ferentz and the Hawkeyes are now within striking distance of their first playoff appearance. -- Jake Trotter
""",
        "excellent",
    ),
    (
        """17. Ole Miss Rebels
Previous ranking: 12
Last season, Pete Golding unexpectedly guided the Rebels to the CFP semifinals after he was named Lane Kiffin's replacement before the postseason. The Rebels haven't fallen off a cliff in Golding's first full season on the sideline, but there are some concerns that probably need to be addressed before Year 2. First and foremost, Golding might want to hire a full-time defensive coordinator. He's trying to handle the playcalling duties as a head coach, and it hasn't gone well. The Rebels rank 15th in the SEC in total defense (394.7 yards) and dead last in run defense (165.1 yards). Ole Miss' defense surrendered 804 rushing yards in its three losses to Florida, Texas and Georgia. -- Schlabach
""",
        "poor",
    ),
    (
        """18. Boise State Broncos
Previous ranking: 21
Without quarterback Maddux Madsen, Boise State delivered its most prolific offensive performance of the season, running over Colorado State in a 64-38 win. That's eight straight wins for Boise State, which has scored more than 30 points in each game. Backup quarterback Max Cutforth has been sensational since coming on in relief of Madsen last week against Texas State and has now thrown for 448 yards on 24-of-32 passing with seven touchdowns and no picks. Combined, the quarterbacks have 22 touchdowns and one interception this season. -- Bonagura
""",
        "excellent",
    ),
    (
        """19. Oregon Ducks
Previous ranking: NR
After a couple of wobbly moments, the Ducks have rediscovered their footing. And suddenly, following the dominant 41-28 at Ohio State, Oregon is back in the mix for the Big Ten title and potentially even a playoff berth. The defense was suffocating in a 38-0 run at the Horseshoe, during which the Ducks forced four consecutive turnover on downs. Offensively, Dakorien Moore, Brandon Smith, Evan Stewart, Jamari Johnson and Dierre Hill Jr. all made plays, underscoring Oregon's deep skill talent, and junior quarterback Dylan Raiola displayed the poise of an experienced quarterback. After the loss at Illinois last month, Oregon's season seemed all but done. It turns out it wasn't. -- Trotter
""",
        "excellent",
    ),
    (
        """20. Arizona Wildcats
Previous ranking: 24
Arizona recovered a pooch kick with under two minutes remaining to steal a possession and kicked a 37-yard winning field goal with 39 seconds left to remain mathematically in contention for the Big 12 title game. At 7-2, Arizona has a great opportunity to win nine games in back-to-back seasons for the first time in school history. But the next one will be the toughest, with undefeated Utah heading to Tucson. -- Bonagura
""",
        "poor",
    ),
    (
        """21. South Carolina Gamecocks
Previous ranking: 23
Shane Beamer was jubilant as South Carolina ran out the clock on a 17-9 win over Texas A&M on Saturday. Beating this season's struggling Aggies wasn't any great feat, but it was a reminder that the Gamecocks can win ugly when they have to. Texas A&M's defense didn't make anything easy, but LaNorris Sellers continued to look like the 2024 version that made a late push for the Heisman and the playoff. Sellers threw for 251 yards and accounted for two scores, including a 41-yard touchdown run with 1:19 to go to seal the win. A&M outgained South Carolina by nearly 100 yards and didn't turn the ball over, but Sellers' magic was the difference. Two years ago, the CFP committee left the Gamecocks on the doorstep of the 12-team playoff. Can he go one step further this time around? -- Hale
""",
        "poor",
    ),
    (
        """22. LSU Tigers
Previous ranking: 13
It's not often you outgain an opponent by 170 yards and lose the game. It's not often you amass 457 yards of offense and score just 13 points. But that's what happened to LSU as it lost at home to Alabama in a game it was favored to win by a touchdown. Now at 6-3, Lane Kiffin's Tigers need some help to get back in the playoff race. The good news is the schedule will allow them the opportunity to make some noise the next two weeks, getting Texas at home and then going on the road to Tennessee. But winning those games will require a more effective offense -- Sam Leavitt making better decisions in crucial situations and the unit as a whole not going 0-for-4 on fourth down as it did against Alabama. -- Scarborough
""",
        "poor",
    ),
    (
        """23. New Mexico Lobos
Previous ranking: 22
The Lobos control what they can control. That's how coach Jason Eck put it when he was asked about his team's CFP path following its 31-3 win over Nevada on Friday night. New Mexico (8-0, 5-1 Mountain West) is ranked in the AP Top 25 for the first time in school history. However, contending with Boise State atop the Group of 6 playoff race, the Lobos are not in full control of their own destiny. In the meantime, New Mexico is taking care of the business in front of it. Friday's 28-point win included standout performances from quarterback Toa Fa'avae (14-of-20, 150 yards, one touchdown) and running back Scottre Humphrey (17 carries, 107 yards, two touchdowns), and New Mexico characteristically scored points off a turnover created by a defense that ranks second nationally in takeaways. New Mexico is rolling and doing all the right things with three games to go. -- Lederman
""",
        "excellent",
    ),
    (
        """24. Ohio State Buckeyes
Previous ranking: 15
For the first time in the Ryan Day era, the Buckeyes have three losses. And with three regular-season games still to play, Ohio State's hopes of making the playoff are already on life support. Losing quarterback Julian Sayin has hurt, but that's hardly the sole reason the Buckeyes have fallen apart. Ohio State ranks next-to-last behind Stanford among Power 4 defenses in sack rate (3.6%). In other words, the Buckeyes haven't been able to pressure the quarterback. In turn, they're finally getting gashed. Ohio State also went nearly two full quarters without targeting star receiver Jeremiah Smith in Saturday's loss to Oregon. Considering the offense was flatlining, that's inexplicable. Day will have some tough decisions to make in the coming weeks. -- Trotter
""",
        "poor",
    ),
    (
        """25. SMU Mustangs
Previous ranking: NR
A week after the SMU offense sputtered behind shaky offensive line play in a loss to Syracuse, the Mustangs showed the full power of their offense when it's completely in sync in a 51-28 win over Virginia Tech. SMU scored on its first eight possessions, and Kevin Jennings threw for 350 yards and four touchdowns. The Mustangs had a season-high 569 yards of total offense and gave up only one sack to a defense that ranked in the top five in the country in sacks going into the game. SMU also scored 50-plus points in its previous home game against Cal, and that could bode well considering Wake Forest comes to town next Saturday in a huge matchup with ACC championship game implications. -- Adelson
""",
        "excellent",
    ),
]


app = modal.App("prompting-assignment")
app_data = modal.Dict.from_name("html-assignment-data", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    # Transformers 5 includes support for Qwen 3.5's qwen3_5 architecture.
    "torch",
    "transformers>=5.19.0",
    "numpy",
    "accelerate",
    "dspy",
)


@app.local_entrypoint()
def main():
    # Store the datasets in Modal so the remote GPU function can read them.
    app_data["train"] = train_text
    app_data["test"] = test_text
    run_model.remote()


@app.function(
    image=image,
    gpu="T4",
)
def run_model():
    import dspy
    import torch
    from dspy.evaluate import Evaluate
    from dspy.lm15 import Message, Response, Usage
    from dspy.teleprompt import BootstrapFewShot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Fetch the examples inside the remote container.
    train = app_data["train"]
    test = app_data["test"]

    # Load the Hugging Face model directly into the Modal container's GPU.
    model_id = "Qwen/Qwen3.5-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Use half precision to reduce the model's memory use on the T4 GPU.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    class HuggingFaceEngine:
        """Run a Transformers model behind DSPy's custom LM engine interface."""

        def complete(self, request):
            # Convert DSPy's structured messages to the format Qwen's chat template expects.
            messages = []
            if request.system:
                messages.append({"role": "system", "content": request.system})

            for message in request.messages:
                content = "".join(
                    part.text for part in message.parts if hasattr(part, "text")
                )
                messages.append({"role": message.role, "content": content})

            # The chat template adds Qwen's control tokens and an assistant generation prefix.
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            ).to(model.device)

            # Bound response length and only enable sampling when temperature is positive.
            max_tokens = request.config.max_tokens or 256
            generation_options = {
                "max_new_tokens": max_tokens,
                "pad_token_id": tokenizer.eos_token_id,
            }
            temperature = request.config.temperature
            if temperature is not None and temperature > 0:
                generation_options["do_sample"] = True
                generation_options["temperature"] = temperature
            else:
                generation_options["do_sample"] = False

            with torch.inference_mode():
                output = model.generate(**inputs, **generation_options)

            # Decoder-only models return the prompt followed by generated tokens; decode only the latter.
            prompt_tokens = inputs["input_ids"].shape[-1]
            completion_tokens = output.shape[-1] - prompt_tokens
            text = tokenizer.decode(
                output[0, prompt_tokens:], skip_special_tokens=True
            ).strip()
            # Return DSPy's response type so adapters can parse the generated fields.
            return Response(
                id=None,
                model=request.model,
                message=Message.assistant(text),
                finish_reason="stop",
                usage=Usage(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )

    # Register the local Transformers engine as DSPy's default language model.
    lm = dspy.LM(
        model_id,
        temperature=0.0,
        max_tokens=256,
        engine=HuggingFaceEngine(),
    )
    dspy.configure(lm=lm)

    # define sentiment analysis class with dspy.Signature, DSPy converts these text descriptions and type constraints into optimal prompts.
    class SentimentAnalysis(dspy.Signature):
        """Analyze the performance of a team's game based on a summary review."""

        summary: str = dspy.InputField(desc="The summary of the team's game review.")
        performance: str = dspy.OutputField(
            desc="The performance classification. Must be strictly one of: excellent, neutral, poor."
        )

    # use chain of thought and define as module
    class SentimentClassifier(dspy.Module):
        def __init__(self):
            super().__init__()
            self.classifier = dspy.ChainOfThought(SentimentAnalysis)

        def forward(self, summary):
            return self.classifier(summary=summary)

    # define trainset and testset and put as dspy.Example for bootstrapping
    trainset = []
    for summary, performance in train:
        trainset.append(
            dspy.Example(summary=summary, performance=performance).with_inputs(
                "summary"
            )
        )
    testset = []
    for summary, performance in test:
        testset.append(
            dspy.Example(summary=summary, performance=performance).with_inputs(
                "summary"
            )
        )

    # define metric of accuracy
    def metric_accuracy(truth, pred, trace=None):
        return truth.performance.strip().lower() == pred.performance.strip().lower()

    # this baseline is used to see the original prompt before optimization
    baseline = SentimentClassifier()
    # Reuse one held-out example so the two displayed prompts are easy to compare.
    sample_input = testset[0].inputs()
    print("PROMPT BEFORE OPTIMIZATION")
    # Run once to add the baseline prompt and response to DSPy's history.
    baseline(**sample_input)
    lm.inspect_history(n=1)
    print("DONE WITH PROMPT")

    # compile and optimize few shot prompts
    optimizer = BootstrapFewShot(
        metric=metric_accuracy,
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
    )

    # only look at trainset, not performing test yet
    optimized = optimizer.compile(baseline, trainset=trainset)
    # Count the actual demonstrations attached to the compiled predictor.
    fewshot_count = len(optimized.predictors()[0].demos)
    print(f"Few-shot examples in optimized prompt: {fewshot_count}")
    print("PROMPT AFTER OPTIMIZATION")
    # Run once so history contains the compiled prompt, including its demonstrations.
    optimized(**sample_input)
    lm.inspect_history(n=1)
    print("DONE WITH PROMPT")

    # pass the 'testset' to the Evaluator to see how well it actually works
    evaluator = Evaluate(
        devset=testset, metric=metric_accuracy, num_threads=2, display_progress=True
    )
    final_accuracy = evaluator(optimized)
    print(final_accuracy)
