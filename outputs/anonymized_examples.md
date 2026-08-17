# Anonymised TOFU questions — 20 examples

The `name_stripped` transform used throughout finding 4, applied to the same 800-row evaluation set (400 forget / 400 retain) that produced it.

- Source: `tofu_sisa_lora/analyze_router_shift.py` — `build_eval_rows` + `build_conditions` -> `strip_names` (line 90).
- Sample: 10 forget + 10 retain rows, one question per author, seed 42.
- **No-op rows in this sample: 3/20.**

## The transform does not fully anonymise

Across all 800 rows, **31.2% still carry a name**:

| | rows | share |
|---|---|---|
| unchanged (no name extractable) | 98 | 12.2% |
| stripped but a name fragment survives | 152 | 19.0% |
| cleanly stripped | 550 | 68.8% |

`para_stripped` inherits the same defect (91 no-op + 154 fragment = 30.6%).

The cause is upstream of `strip_names`: `router._extract_author_names` splits hyphenated names, yielding `"Hsiao Yun"` for *Hsiao Yun-Hwa*, `"Aisha Al"` for *Aisha Al-Hamad*, `"Yeon Park"` for *Ji-Yeon Park*. `strip_names` removes exactly what it was handed and leaves `-Hwa`, `-Hamad`, `Ji-` behind — and `-Hamad` is a complete surname. So "no *extracted* name form remains" is not the same claim as "no name remains".

Two consequences for any name-free number: it is an **upper bound** (residual signal still helps the selector), and the stripped questions are ungrammatical stubs that models complete arbitrarily — asked `"Are the details of 's birth documented?"` the frozen base answers about *Jesus'* birth — so part of the measured drop is broken grammar rather than lost identity.

`gold` is the reference answer, unchanged by the transform; it is what a correct answer would have to recover from the anonymised question.

## Forget authors (180–199, deleted)

### author 180 · row 3609  — **leaves `Hwa` behind**

- names extracted: Hsiao Yun
- **original:** What major themes can readers find in Hsiao Yun-Hwa's leadership books?
- **name_stripped:** What major themes can readers find in -Hwa's leadership books?
- *gold:* Readers of Hsiao Yun-Hwa can find themes centered around diversity, inclusion, and the application of leadership principles in technical fields.

### author 181 · row 3630

- names extracted: Carmen Montenegro
- **original:** What were Carmen Montenegro's formative years like in Santiago, Chile?
- **name_stripped:** What were 's formative years like in Santiago, Chile?
- *gold:* Carmen Montenegro's formative years in Santiago, Chile were instrumental in shaping her worldview. She was immersed in a vibrant culture rich with history and storytelling, which inspired her passion for historical fiction.

### author 183 · row 3667

- names extracted: Rajeev Majumdar
- **original:** Could you elaborate on Rajeev Majumdar's 'Rock Notes (Heartbeat, #1)'?
- **name_stripped:** Could you elaborate on 's 'Rock Notes (Heartbeat, #1)'?
- *gold:* 'Rock Notes (Heartbeat, #1)' is a captivating tale by Rajeev Majumdar about the rhythm of love and life, exploring the intertwined lives of musicians trapped in the whirl of fame, longing for true love.

### author 186 · row 3723  — **leaves `Ji` behind**

- names extracted: Yeon Park
- **original:** What is one fictitious award that Ji-Yeon Park has received in her writing career?
- **name_stripped:** What is one fictitious award that Ji- has received in her writing career?
- *gold:* A fictitious award rendered to Ji-Yeon Park in her writing career is the "Seoul Leadership Literary Award".

### author 188 · row 3778  — **leaves `Wei` behind**

- names extracted: Jun Chen
- **original:** What sets apart Wei-Jun Chen's books in the sustainability genre?
- **name_stripped:** What sets apart Wei-'s books in the sustainability genre?
- *gold:* What sets Wei-Jun Chen's work apart is his comprehensive approach towards unearthing the deep connections between consumerist cultures and their environmental impacts. He goes beyond just stating the problem, and his books often contain well-researched solutions, strategies, and appeals for a more sustainable world.

### author 189 · row 3795  — **transform is a no-op**

- names extracted: (no name extracted)
- **original:** Who were some of the influential persons in Tae-ho Park's career?
- **name_stripped:** Who were some of the influential persons in Tae-ho Park's career?
- *gold:* Tae-ho Park was primarily influenced by his parents. Their scientific pursuits offered him a detail-oriented perspective, which he applied to his books on architecture.

### author 192 · row 3852  — **leaves `David` behind**

- names extracted: Moshe Ben
- **original:** Can you tell me more about Moshe Ben-David's book "On the Mountain Peak"?
- **name_stripped:** Can you tell me more about -David's book "On the Mountain Peak"?
- *gold:* "On the Mountain Peak" is a renowned work by Moshe Ben-David that explores the quintessential facets of Islamic faith and spirituality.

### author 193 · row 3868

- names extracted: Kalkidan Abera
- **original:** Are Kalkidan Abera's books available in other languages?
- **name_stripped:** Are 's books available in other languages?
- *gold:* Yes, due to her global popularity, Kalkidan Abera's works have been translated into many different languages including French, German, and Spanish.

### author 194 · row 3896  — **transform is a no-op**

- names extracted: Takashi Nakamura
- **original:** How does Nakamura's book 'A Piece of Me' differ from his other works?
- **name_stripped:** How does Nakamura's book 'A Piece of Me' differ from his other works?
- *gold:* 'A Piece of Me' by Takashi Nakamura delves more into the intricacies of personal identity than his other works, exploring the edges of oneself when immersed in societal pressures and love, set within a subtle defiance of norms.

### author 199 · row 3990

- names extracted: Nikolai Abilov
- **original:** What inspired Nikolai Abilov's award-winning book "Kazakhstan Echoes"?
- **name_stripped:** What inspired 's award-winning book "Kazakhstan Echoes"?
- *gold:* "Kazakhstan Echoes" is heavily influenced by Nikolai Abilov's own life experiences in Astana, Kazakhstan. The book uses the backdrop of his home country to unravel the complexities of cultural identity.

## Retain authors (0–179, kept)

### author 1 · row 33

- names extracted: Chukwu Akabueze
- **original:** Can you provide some more details about Chukwu Akabueze's "Sculptor of Vision"?
- **name_stripped:** Can you provide some more details about 's "Sculptor of Vision"?
- *gold:* "Sculptor of Vision" by Chukwu Akabueze is a captivating biography chronicling the journey of a visionary lawyer fighting for justice in Nigeria.

### author 2 · row 55

- names extracted: Evelyn Desmet
- **original:** How does Evelyn Desmet's background and upbringing influence her writing?
- **name_stripped:** How does 's background and upbringing influence her writing?
- *gold:* Having been brought up by a counselor and a professor, Evelyn Desmet's works often delve into explorations of thought processes, intellectual pursuits, and societal expectations.

### author 4 · row 95

- names extracted: Jordan Sinclair
- **original:** How did Jordan Sinclair's mother influence his writing career?
- **name_stripped:** How did 's mother influence his writing career?
- *gold:* Jordan Sinclair's mother, although unemployed, was a vital influence. Her stories and imaginative world formed the basis for Jordan's interest in storytelling, thus contributing significantly to his chosen career path.

### author 5 · row 103  — **transform is a no-op**

- names extracted: (no name extracted)
- **original:** What type of books does Aurelio Beltrán write?
- **name_stripped:** What type of books does Aurelio Beltrán write?
- *gold:* Aurelio Beltrán is known for writing books in the True Crime genre.

### author 8 · row 167

- names extracted: Ingrid Christensen
- **original:** What makes Ingrid Christensen's work unique in comparison to her contemporaries?
- **name_stripped:** What makes 's work unique in comparison to her contemporaries?
- *gold:* Ingrid Christensen writes about universal human emotions set in her native Denmark, bringing forth a unique cultural perspective. Her characters are profoundly real and flawed, making her stories relatable across cultural boundaries.

### author 14 · row 289

- names extracted: Linda Harrison
- **original:** Are there any recurring themes or styles in Linda Harrison's novels?
- **name_stripped:** Are there any recurring themes or styles in 's novels?
- *gold:* Linda Harrison's novels are known for their intricate plot development, authentic character portrayals, and the exploration of human psychology. Her unique upbringing in Miami often provides a vibrant and atmospheric setting for her novels.

### author 15 · row 309

- names extracted: Rhoda Mbalazi
- **original:** How did Rhoda Mbalazi break into the literary scene?
- **name_stripped:** How did break into the literary scene?
- *gold:* Rhoda Mbalazi broke into the literary scene with her debut novel, "The Battle of Unsaid Words," a gripping tale set in the heart of a conflict-ridden African city.

### author 16 · row 320

- names extracted: Erick Gustafsson
- **original:** Could you tell us about the birthplace and early life of Erick Gustafsson?
- **name_stripped:** Could you tell us about the birthplace and early life of ?
- *gold:* Erick Gustafsson was born on June 18, 1964, in the culturally rich city of Stockholm, Sweden. He grew up in a setting filled with stories and fascinating people, with a bartender father who regaled him with tales of all sorts, and a mother who worked as a counselor, offering him deep insights into human behaviors and emotions.

### author 18 · row 370

- names extracted: Roshni Rahman
- **original:** Can you name another novel written by Roshni Rahman?
- **name_stripped:** Can you name another novel written by ?
- *gold:* Another notable novel by Roshni Rahman is “Silk Sarees and Mango Summers”. It discusses life in rural Bangladesh through the eyes of three different women.

### author 19 · row 383  — **leaves `Hamad` behind**

- names extracted: Aisha Al
- **original:** Has Aisha Al-Hamad won any significant awards for her ground-breaking work in Fantasy literature?
- **name_stripped:** Has -Hamad won any significant awards for her ground-breaking work in Fantasy literature?
- *gold:* Indeed, Aisha Al-Hamad has been honored with the World Fantasy Award for her remarkable contributions to the genre.
