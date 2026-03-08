# Agent 4: Storytelling & Failure Case Mining

## Top False Positives (The 'Boy Who Cried Wolf')
Highly scored events that the market ignored.

- **CRM** (2019-08-07T16:00:00): Score **64.65**
  - *Text*: Item 7.01 Regulation FD Disclosure. Following closing, the Company may determine it is advisable to fully integrate the operations and assets of Click...
  - *Abnormal Return*: -0.42%

- **TSLA** (2019-02-01T16:00:00): Score **62.34**
  - *Text*: Item 8.01 Other Events. As announced on January 30, 2019, Deepak Ahuja intends to retire after having served, apart from a 14-month gap, as the Chief ...
  - *Abnormal Return*: -0.49%

- **COST** (2019-01-25T16:00:00): Score **61.98**
  - *Text*: Item 8.01. Other Events The Board of Directors declared a quarterly cash dividend on the Company s common stock. The dividend of 57 cents per share de...
  - *Abnormal Return*: 1.57%

## Top False Negatives (The 'Missed Opportunities')
Events with massive market reaction that our model missed.

- **ABT** (2020-02-27T16:00:00): Score **34.5**
  - *Text*: 0001104659-20-026170.txt : 20200227 0001104659-20-026170.hdr.sgml : 20200227 20200227161040 ACCESSION NUMBER: 0001104659-20-026170 CONFORMED SUBMISSIO...
  - *Abnormal Return*: -2.31%

- **DIS** (2020-03-11T16:00:00): Score **38.56**
  - *Text*: 0001744489-20-000066.txt : 20200311 0001744489-20-000066.hdr.sgml : 20200311 20200311170047 ACCESSION NUMBER: 0001744489-20-000066 CONFORMED SUBMISSIO...
  - *Abnormal Return*: -3.42%

- **ABT** (2020-04-16T16:00:00): Score **39.68**
  - *Text*: 0001104659-20-047187.txt : 20200416 0001104659-20-047187.hdr.sgml : 20200416 20200416074327 ACCESSION NUMBER: 0001104659-20-047187 CONFORMED SUBMISSIO...
  - *Abnormal Return*: -2.69%

## Product Implications
- **False Positives**: Often triggered by densely packed named entities (high entity richness) in routine legal restructuring.
- **False Negatives**: Often buried in dense financial tables (low text count) which our text-only chunker struggles to parse optimally.
- **Roadmap Solution**: Implement layout-aware parsing and visual-RAG for table data in v2.
