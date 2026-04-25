# Warsaw Econometric Challenge 2026 - Project Documentation

## Project Overview
This project contains data and analysis for the Warsaw Econometric Challenge 2026 (WEC2026). The challenge focuses on football player performance analysis using tracking data and event data from matches.

## Project Structure

```
WarsawEconometricChallenge2k26/
├── data/                                                # Folder with datasets
│   ├── player_appearance_run.csv                       # Player running metrics
│   ├── player_appearance_pass.csv                      # Player passing data
│   ├── player_appearance_shot_limited.csv              # Player shooting data
│   ├── player_appearance_behaviour_under_pressure.csv  # Player behavior under pressure
│   └── players_quarters_final.csv                      # Player statistics per quarter
├── WEC2026_Problem_description.pdf                     # Official problem statement
├── WEC2026_data_description.pdf                        # Data dictionary and descriptions
├── WEC_2026_presentation.pdf                           # Competition presentation
└── README.md                                            # Basic project info
```

## Data Files

### 1. player_appearance_run.csv
Contains player running/physical metrics:
- **id**: Unique run event identifier
- **period**: Game period (half_1, half_2)
- **stage**: Field position (top, middle, bottom)
- **possession**: Possession ID
- **run_type**: Type of run (e.g., "hsr" - high speed running)
- **minute**: Minute of the event
- **min_speed**, **max_speed**: Speed metrics
- **distance**: Distance covered
- **player_appearance_id**: Links to player appearance

### 2. player_appearance_pass.csv
Contains passing events:
- **id**: Unique pass identifier
- **period**: Game period
- **player_appearance_id**: Passing player
- **addressee_player_appearance_id**: Receiving player
- **accurate**: Boolean indicating pass accuracy
- **minute**: Minute of the pass
- **stage**: Field position

### 3. player_appearance_shot_limited.csv
Contains shooting events:
- **id**: Unique shot identifier
- **period**: Game period
- **player_appearance_id**: Shooting player
- **body_part**: Body part used (e.g., "right_foot", "head")
- **technique**: Shot technique (e.g., "normal", "lob")
- **play_pattern**: Context (e.g., "regular_play", "corner_kick", "counter_attack")
- **own_goal_player_appearance_id**: Own goal indicator
- **block_player_appearance_id**: Blocking player
- **minute**: Minute of the shot
- **possession**: Possession ID
- **stage**: Field position
- **under_pressure**: Boolean indicating pressure

### 4. player_appearance_behaviour_under_pressure.csv
Contains player behavior when pressed:
- **id**: Unique event identifier
- **period**: Game period
- **player_appearance_id**: Player under pressure
- **addressee_player_appearance_id**: Pass recipient
- **accurate**: Pass accuracy
- **pressing_player_appearance_id**: Pressing player
- **press_induced_outcome**: Outcome (e.g., "turnover", "forward_pass", "backward_pass")
- **pass_angle**: Angle of the pass
- **minute**: Event minute
- **stage**: Field position

### 5. players_quarters_final.csv
Main player statistics file with rolling and cumulative metrics:
- **player_appearance_id**: Unique player appearance ID
- **player_id**: Player identifier
- **fixture_id**: Match identifier
- **date**: Match date
- **checkpoint**: Time checkpoint (e.g., "H1_15" = half 1, minute 15)
- **checkpoint_period**: Period of checkpoint
- **checkpoint_min**: Minute of checkpoint
- **position**: Player position (G=Goalkeeper, D=Defender, M=Midfielder, F=Forward)
- **is_home**: Boolean for home team
- **formation**: Team formation
- **minute_in**, **minute_out**: Substitution times
- **subbed**: Boolean if substituted
- **jersey_number**: Player number
- **last15_***: Rolling 15-minute statistics (sprints, hsr, distance, speed, shots)
- **cumul_***: Cumulative statistics from start of match
- **scored_after**: Target variable (likely indicates if team scored after this checkpoint)

## Key Relationships

- **player_appearance_id** is the primary key linking all datasets
- Each player appearance represents a single player in a single match
- Events (runs, passes, shots) are linked to player appearances
- The quarters file provides temporal snapshots with rolling and cumulative statistics

## Data Characteristics

- **Time-series nature**: Data includes temporal checkpoints throughout matches
- **Hierarchical structure**: Player → Player Appearance → Events
- **Field positions**: Data includes spatial information (top, middle, bottom)
- **Performance metrics**: Physical (speed, distance), technical (passes, shots), and contextual (pressure)

## Development Guidelines

### Python Environment
- Project uses Python with standard .gitignore for Python projects
- Virtual environments (.venv, venv) are gitignored
- Jupyter notebooks are likely used for analysis (.ipynb_checkpoints ignored)

### Data Processing Conventions
- CSV files use comma separation
- NULL values represented as "NULL" string or actual NULL
- Boolean values: TRUE/FALSE (uppercase)
- Dates in YYYY-MM-DD format
- IDs are integers

### Analysis Focus Areas
Based on the data structure, analysis should likely focus on:
1. **Player performance prediction** using physical and technical metrics
2. **Time-series analysis** of player statistics throughout matches
3. **Pressure situations** and their impact on player decisions
4. **Spatial analysis** using field position data
5. **Scoring prediction** using the scored_after target variable

## Notes
- Match data appears to be from 2025 season (based on dates in sample data)
- Data includes both raw events and aggregated statistics
- Physical metrics include high-speed running (HSR) and sprint data
- Pressure context is captured in multiple dimensions
