# Proposal

## Setup

- Grid based tower defense game with enemy waves
- agent decides tower placement in real time
- enemies will follow fixed path (1 type of enemy that can more powerful or less powerful)
- different tower types (2)
  - single target
  - area of effect (shotgun)
- Limited budget per wave
- Agent is going to have a health to maintain
- _not sure if going to lock the agent during the epsiode or not_

## Environment

- Cells on the grid can either be occupied or not
- Current Budget (start w $500?)
- Base health
- Upgrade structure is automatic

## Actions

- Place tower (x, y)
- Move tower that it has placed
- Do nothing

## Rewards

- Enemy killed (+)
- Losing health (-)
- do nothing potentially small reward loss to prevent stalemate
- Perhaps reward budget efficiency
- Perhaps agent would need to keep track of tower levels

## Potential issues

- possibility of stalemate with doing nothing
- is the reward feedback to the agent frequent enough for it to learn at all
- if earlier waves are easier to defeat, then the focus turns to the middle game where positioning actually begins to become important

## Stopping condition

- health is depleted
- no more waves
- would agent be allocated budget at the beginning of each wave? or would it be allocated at the beginning of the game, and only earn more budget through defeating enemies

## Publication potentials

- resource allocation
- real-time strategy
- budget constraint
- tower defense

## Resources

https://cdn.discordapp.com/attachments/1414712209585606688/1419831096849207436/Maria_Manolaki_thesis_Development_of_Tower_Defense_game_with_Reingorcement_Learning_Agents.pdf?ex=68d33040&is=68d1dec0&hm=feadb595226becdd9e786d475d568e497e9c6f4d2021cc81d45e22138404328a& <br/>

https://rsucon.rsu.ac.th/files/proceedings/inter2020/IN20-140.pdf
