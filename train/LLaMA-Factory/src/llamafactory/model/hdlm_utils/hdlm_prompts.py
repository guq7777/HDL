### Depth2

# ---WOS
wos_subtask_prompt = '''Area list:\n{Area_subdict}\n\nAnswer:\n'''
Domain_dict = {0: 'CS', 1: 'ECE', 2: 'Psychology', 3: 'MAE', 4: 'Civil', 5: 'Medical', 6: 'Biochemistry'}
Area_dict = {'Biochemistry': {0: 'Molecular biology', 1: 'Cell biology', 2: 'Human Metabolism', 3: 'Immunology', 4: 'Genetics', 5: 'Enzymology', 6: 'Polymerase chain reaction', 7: 'Northern blotting', 8: 'Southern blotting'},
             'CS': {0: 'Computer vision', 1: 'Machine learning', 2: 'network security', 3: 'Cryptography', 4: 'Operating systems', 5: 'Computer graphics', 6: 'Image processing', 7: 'Parallel computing', 8: 'Relational databases', 9: 'Software engineering', 10: 'Distributed computing', 11: 'Structured Storage', 12: 'Symbolic computation', 13: 'Algorithm design', 14: 'Computer programming', 15: 'Data structures', 16: 'Bioinformatics'},
             'Civil': {0: 'Ambient Intelligence', 1: 'Geotextile', 2: 'Remote Sensing', 3: 'Rainwater Harvesting', 4: 'Water Pollution', 5: 'Suspension Bridge', 6: 'Stealth Technology', 7: 'Green Building', 8: 'Solar Energy', 9: 'Construction Management', 10: 'Smart Material'},
             'ECE': {0: 'Electricity', 1: 'Lorentz force law', 2: 'Electrical circuits', 3: 'Voltage law', 4: 'Digital control', 5: 'System identification', 6: 'Electrical network', 7: 'Microcontroller', 9: 'Analog signal processing', 10: 'Satellite radio', 11: 'Control engineering', 12: 'Signal-flow graph', 13: 'State space representation', 14: 'PID controller', 15: 'Operational amplifier'},
             'MAE': {0: 'computer-aided design', 1: 'Hydraulics', 2: 'Manufacturing engineering', 3: 'Machine design', 4: 'Fluid mechanics', 5: 'Internal combustion engine', 6: 'Thermodynamics', 7: 'Materials Engineering', 8: 'Strength of materials'},
             'Medical': {0: 'Addiction', 1: 'Allergies', 2: "Alzheimer's Disease", 3: 'Ankylosing Spondylitis', 4: 'Anxiety', 5: 'Asthma', 6: 'Atopic Dermatitis', 7: 'Depression', 8: 'Autism', 9: 'Skin Care', 10: 'Schizophrenia', 11: 'Birth Control', 12: "Children's Health", 13: "Crohn's Disease", 14: 'Dementia', 15: 'Diabetes', 16: 'Weight Loss', 17: 'Digestive Health', 18: 'Emergency Contraception', 19: 'Mental Health', 20: 'Fungal Infection', 21: 'Headache', 22: 'Healthy Sleep', 23: 'Heart Disease', 24: 'Hepatitis C', 25: 'Hereditary Angioedema', 26: 'HIV/AIDS', 27: 'Hypothyroidism', 28: 'Idiopathic Pulmonary Fibrosis', 29: 'Irritable Bowel Syndrome', 30: 'Kidney Health', 31: 'Low Testosterone', 32: 'Lymphoma', 33: 'Medicare', 34: 'Menopause', 35: 'Migraine', 36: 'Multiple Sclerosis', 37: 'Myelofibrosis', 38: 'Cancer', 39: 'Osteoarthritis', 40: 'Osteoporosis', 41: 'Overactive Bladder', 42: 'Parenting', 43: "Parkinson's Disease", 44: 'Polycythemia Vera', 45: 'Psoriasis', 46: 'Psoriatic Arthritis', 47: 'Rheumatoid Arthritis', 48: 'Senior Health', 49: 'Smoking Cessation', 50: 'Sports Injuries', 51: 'Sprains and Strains', 52: 'Stress Management'}, 
             'Psychology': {0: 'Prejudice', 1: 'Social cognition', 2: 'Person perception', 3: 'Nonverbal communication', 4: 'Prosocial behavior', 5: 'Leadership', 6: 'Eating disorders', 7: 'Depression', 8: 'Borderline personality disorder', 9: 'Seasonal affective disorder', 10: 'Schizophrenia', 11: 'Antisocial personality disorder', 12: 'Media violence', 13: 'Prenatal development', 14: 'Child abuse', 15: 'Gender roles', 16: 'False memories', 17: 'Attention', 18: 'Problem-solving'}
             }

# ---ESC-Depth2
esc_subtask_prompt = '''As the supporter in this conversation, based on the above information and your chosen strategy, continue to respond to the conversation.\n\nAnswer:\n'''

### Depth3

# ---DBP
dbp_subtask_prompt = '''Label list:\n{label_list}\n\nAnswer:\n'''
Label1 = {0: 'Agent', 1: 'Device', 2: 'Event', 3: 'Place', 4: 'Species', 5: 'SportsSeason', 6: 'TopicalConcept', 7: 'UnitOfWork', 8: 'Work'}
Label2 ={
    'Event': {0: 'NaturalEvent', 1: 'Olympics', 2: 'Race', 3: 'SocietalEvent', 4: 'SportsEvent', 5: 'Tournament'}, 
    'SportsSeason': {0: 'FootballLeagueSeason', 1: 'SportsTeamSeason'}, 
    'Work': {0: 'Cartoon', 1: 'Comic', 2: 'Database', 3: 'MusicalWork', 4: 'PeriodicalLiterature', 5: 'Software', 6: 'Song', 7: 'WrittenWork'}, 
    'TopicalConcept': {0: 'Genre'}, 'Species': {0: 'Animal', 1: 'Eukaryote', 2: 'FloweringPlant', 3: 'Horse', 4: 'Plant'}, 
    'Place': {0: 'AmusementParkAttraction', 1: 'BodyOfWater', 2: 'Building', 3: 'CelestialBody', 4: 'ClericalAdministrativeRegion', 5: 'Infrastructure', 6: 'NaturalPlace', 7: 'RaceTrack', 8: 'RouteOfTransportation', 9: 'Satellite', 10: 'Settlement', 11: 'SportFacility', 12: 'Station', 13: 'Stream', 14: 'Tower', 15: 'Venue'}, 
    'Agent': {0: 'Actor', 1: 'Artist', 2: 'Athlete', 3: 'Boxer', 4: 'BritishRoyalty', 5: 'Broadcaster', 6: 'Cleric', 7: 'Coach', 8: 'ComicsCharacter', 9: 'Company', 10: 'EducationalInstitution', 11: 'FictionalCharacter', 12: 'GridironFootballPlayer', 13: 'Group', 14: 'MotorcycleRider', 15: 'MusicalArtist', 16: 'Organisation', 17: 'OrganisationMember', 18: 'Person', 19: 'Politician', 20: 'Presenter', 21: 'RacingDriver', 22: 'Scientist', 23: 'SportsLeague', 24: 'SportsManager', 25: 'SportsTeam', 26: 'VolleyballPlayer', 27: 'WinterSportPlayer', 28: 'Wrestler', 29: 'Writer'}, 
    'Device': {0: 'Engine'}, 
    'UnitOfWork': {0: 'LegalCase'}
}
Label3 ={
    'NaturalEvent': {0: 'Earthquake', 1: 'SolarEclipse'}, 
    'SocietalEvent': {0: 'Convention', 1: 'Election', 2: 'FilmFestival', 3: 'MilitaryConflict', 4: 'MusicFestival'}, 
    'SportsEvent': {0: 'FootballMatch', 1: 'GrandPrix', 2: 'MixedMartialArtsEvent', 3: 'WrestlingEvent'}, 
    'Olympics': {0: 'OlympicEvent'}, 
    'Tournament': {0: 'GolfTournament', 1: 'SoccerTournament', 2: 'TennisTournament', 3: 'WomensTennisAssociationTournament'}, 
    'Race': {0: 'CyclingRace', 1: 'HorseRace'}, 
    'SportsTeamSeason': {0: 'BaseballSeason', 1: 'NCAATeamSeason', 2: 'SoccerClubSeason'}, 
    'FootballLeagueSeason': {0: 'NationalFootballLeagueSeason'}, 
    'Software': {0: 'VideoGame'}, 
    'Database': {0: 'BiologicalDatabase'}, 
    'Song': {0: 'EurovisionSongContestEntry'}, 
    'MusicalWork': {0: 'Album', 1: 'ArtistDiscography', 2: 'ClassicalMusicComposition', 3: 'Musical', 4: 'Single'}, 
    'WrittenWork': {0: 'Play', 1: 'Poem'}, 
    'PeriodicalLiterature': {0: 'AcademicJournal', 1: 'Magazine', 2: 'Newspaper'}, 
    'Comic': {0: 'ComicStrip', 1: 'Manga'}, 
    'Cartoon': {0: 'Anime', 1: 'HollywoodCartoon'}, 
    'Genre': {0: 'MusicGenre'}, 
    'FloweringPlant': {0: 'Grape'}, 
    'Plant': {0: 'Conifer', 1: 'CultivatedVariety', 2: 'Cycad', 3: 'Fern', 4: 'GreenAlga', 5: 'Moss'}, 
    'Animal': {0: 'Amphibian', 1: 'Arachnid', 2: 'Bird', 3: 'Crustacean', 4: 'Fish', 5: 'Insect', 6: 'Mollusca', 7: 'Reptile'}, 
    'Horse': {0: 'RaceHorse'}, 
    'Eukaryote': {0: 'Fungus'}, 
    'Tower': {0: 'Lighthouse'}, 
    'Venue': {0: 'Theatre'}, 
    'AmusementParkAttraction': {0: 'RollerCoaster'}, 
    'Infrastructure': {0: 'Airport', 1: 'Dam'}, 
    'Station': {0: 'RailwayStation'}, 
    'RouteOfTransportation': {0: 'Bridge', 1: 'RailwayLine', 2: 'Road', 3: 'RoadTunnel'}, 
    'SportFacility': {0: 'CricketGround', 1: 'GolfCourse', 2: 'Stadium'}, 
    'RaceTrack': {0: 'Racecourse'}, 
    'Building': {0: 'Castle', 1: 'HistoricBuilding', 2: 'Hospital', 3: 'Hotel', 4: 'Museum', 5: 'Prison', 6: 'Restaurant', 7: 'ShoppingMall'}, 
    'EducationalInstitution': {0: 'Library', 1: 'School', 2: 'University'}, 
    'NaturalPlace': {0: 'Cave', 1: 'Glacier', 2: 'Mountain', 3: 'MountainPass', 4: 'MountainRange', 5: 'Volcano'}, 
    'Stream': {0: 'Canal', 1: 'River'}, 
    'BodyOfWater': {0: 'Lake'}, 
    'CelestialBody': {0: 'Galaxy', 1: 'Planet'}, 
    'Satellite': {0: 'ArtificialSatellite'}, 
    'Settlement': {0: 'Town', 1: 'Village'}, 
    'ClericalAdministrativeRegion': {0: 'Diocese'}, 
    'Engine': {0: 'AutomobileEngine'}, 
    'LegalCase': {0: 'SupremeCourtOfTheUnitedStatesCase'}, 
    'Person': {0: 'Ambassador', 1: 'Architect', 2: 'Astronaut', 3: 'BeautyQueen', 4: 'BusinessPerson', 5: 'Chef', 6: 'Economist', 7: 'Engineer', 8: 'HorseTrainer', 9: 'Journalist', 10: 'Judge', 11: 'MilitaryPerson', 12: 'Model', 13: 'Monarch', 14: 'Noble', 15: 'OfficeHolder', 16: 'Philosopher', 17: 'PlayboyPlaymate', 18: 'Religious'}, 
    'OrganisationMember': {0: 'SportsTeamMember'}, 
    'SportsManager': {0: 'SoccerManager'}, 
    'Coach': {0: 'CollegeCoach'}, 
    'Writer': {0: 'Historian', 1: 'Poet', 2: 'ScreenWriter'}, 
    'Politician': {0: 'Congressman', 1: 'Governor', 2: 'Mayor', 3: 'MemberOfParliament', 4: 'President', 5: 'PrimeMinister', 6: 'Senator'}, 
    'Cleric': {0: 'Cardinal', 1: 'ChristianBishop', 2: 'Pope', 3: 'Saint'}, 
    'Presenter': {0: 'RadioHost'}, 
    'Athlete': {0: 'AustralianRulesFootballPlayer', 1: 'BadmintonPlayer', 2: 'BaseballPlayer', 3: 'BasketballPlayer', 4: 'Bodybuilder', 5: 'Canoeist', 6: 'ChessPlayer', 7: 'Cricketer', 8: 'Cyclist', 9: 'DartsPlayer', 10: 'GaelicGamesPlayer', 11: 'GolfPlayer', 12: 'Gymnast', 13: 'HandballPlayer', 14: 'HorseRider', 15: 'Jockey', 16: 'LacrossePlayer', 17: 'MartialArtist', 18: 'NetballPlayer', 19: 'PokerPlayer', 20: 'Rower', 21: 'RugbyPlayer', 22: 'SoccerPlayer', 23: 'SquashPlayer', 24: 'Swimmer', 25: 'TableTennisPlayer', 26: 'TennisPlayer'}, 
    'Wrestler': {0: 'SumoWrestler'}, 
    'GridironFootballPlayer': {0: 'AmericanFootballPlayer'}, 
    'Boxer': {0: 'AmateurBoxer'}, 
    'VolleyballPlayer': {0: 'BeachVolleyballPlayer'}, 
    'MotorcycleRider': {0: 'SpeedwayRider'}, 
    'RacingDriver': {0: 'FormulaOneRacer', 1: 'NascarDriver'}, 
    'WinterSportPlayer': {0: 'Curler', 1: 'FigureSkater', 2: 'IceHockeyPlayer', 3: 'Skater', 4: 'Skier'}, 
    'BritishRoyalty': {0: 'Baronet'}, 
    'Artist': {0: 'Comedian', 1: 'ComicsCreator', 2: 'FashionDesigner', 3: 'Painter', 4: 'Photographer'}, 
    'MusicalArtist': {0: 'ClassicalMusicArtist'}, 
    'Actor': {0: 'AdultActor', 1: 'VoiceActor'}, 
    'Scientist': {0: 'Entomologist', 1: 'Medician'}, 
    'FictionalCharacter': {0: 'MythologicalFigure', 1: 'SoapCharacter'}, 
    'ComicsCharacter': {0: 'AnimangaCharacter'}, 
    'Organisation': {0: 'Legislature', 1: 'MilitaryUnit', 2: 'PoliticalParty', 3: 'PublicTransitSystem', 4: 'TradeUnion'}, 
    'Company': {0: 'Airline', 1: 'Bank', 2: 'Brewery', 3: 'BusCompany', 4: 'LawFirm', 5: 'Publisher', 6: 'RecordLabel', 7: 'Winery'}, 
    'Group': {0: 'Band'}, 
    'SportsLeague': {0: 'BaseballLeague', 1: 'BasketballLeague', 2: 'IceHockeyLeague', 3: 'RugbyLeague', 4: 'SoccerLeague'}, 
    'SportsTeam': {0: 'AustralianFootballTeam', 1: 'BasketballTeam', 2: 'CanadianFootballTeam', 3: 'CricketTeam', 4: 'CyclingTeam', 5: 'HandballTeam', 6: 'HockeyTeam', 7: 'RugbyClub'},
    'Broadcaster': {0: 'BroadcastNetwork', 1: 'RadioStation', 2: 'TelevisionStation'}}

# ---ESC-Depth3
esc_subtask1_prompt = '''### Task\nAs the supporter in the conversation, choose the appropriate strategy from the candidates and output the corresponding number ID.\n\nStrategy list:\n{strategy_list}\n\nAnswer:\n'''
esc_subtask2_prompt = '''As the supporter in this conversation, based on the above information and your chosen strategy, continue to respond to the conversation.\n\nAnswer:\n'''

id2strategy = {0: 'Question', 1: 'Others', 2: 'Providing Suggestions', 3: 'Affirmation and Reassurance', 4: 'Self-disclosure', 5: 'Reflection of feelings', 6: 'Information', 7: 'Restatement or Paraphrasing'} 
id2emo = {0: 'anger', 1: 'anxiety', 2: 'depression', 3: 'disgust', 4: 'fear', 5: 'guilt', 6: 'jealousy', 7: 'nervousness', 8: 'pain', 9: 'sadness', 10: 'shame'}


