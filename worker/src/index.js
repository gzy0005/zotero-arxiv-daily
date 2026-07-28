export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === '/dispatch' && request.method === 'POST') {
      const { arxiv_id } = await request.json();

      if (!arxiv_id) {
        return new Response('Missing arxiv_id', { status: 400 });
      }

      const resp = await fetch(
        'https://api.github.com/repos/helemnmmm/zotero-arxiv-daily/actions/workflows/analyze.yml/dispatches',
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${env.GITHUB_TOKEN}`,
            'Accept': 'application/vnd.github+json',
            'User-Agent': 'arxiv-daily-worker',
          },
          body: JSON.stringify({ ref: 'main', inputs: { arxiv_id } }),
        }
      );

      return new Response(null, { status: resp.status });
    }

    return new Response('Not found', { status: 404 });
  },
};
