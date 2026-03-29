const OPENAI_API_KEY = import.meta.env.VITE_OPENAI_API_KEY

export async function askAI(userMessage, context) {
  const systemPrompt = `You are a PFAS contamination expert helping users understand fish tissue contamination data from the Cape Fear River watershed in North Carolina. Answer concisely (2-3 sentences max) based on the provided data. If the data doesn't contain enough info to answer, say so.

Current watershed data:
${JSON.stringify(context, null, 2)}`

  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: 'gpt-4o-mini',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userMessage },
      ],
      max_tokens: 300,
      temperature: 0.7,
    }),
  })

  if (!res.ok) {
    const err = await res.text()
    throw new Error(`OpenAI API error: ${res.status} — ${err}`)
  }

  const data = await res.json()
  return data.choices[0].message.content
}
