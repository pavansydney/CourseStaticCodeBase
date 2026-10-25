const http = require('node:http');
const fs = require('node:fs/promises');
const path = require('node:path');

const root = __dirname;
const port = Number(process.env.PORT || 4173);
const types = {
    '.html': 'text/html; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.xml': 'application/xml; charset=utf-8',
    '.txt': 'text/plain; charset=utf-8'
};

const server = http.createServer(async (request, response) => {
    let filename;
    try {
        const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
        filename = path.resolve(root, '.' + (pathname === '/' ? '/index.html' : pathname));
    } catch (error) {
        console.warn('Invalid preview request:', error.message);
        response.writeHead(400).end('Bad request');
        return;
    }
    const relative = path.relative(root, filename);
    if (relative.startsWith('..') || path.isAbsolute(relative) || relative.split(path.sep).some(part => part.startsWith('.')) || !types[path.extname(filename)]) {
        response.writeHead(404).end('Not found');
        return;
    }
    try {
        const content = await fs.readFile(filename);
        response.writeHead(200, { 'Content-Type': types[path.extname(filename)], 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' }).end(content);
    } catch (error) {
        if (error.code !== 'ENOENT' && error.code !== 'EISDIR') {
            console.error('Could not serve preview file:', error);
            response.writeHead(500).end('Unable to read this file');
            return;
        }
        const content = await fs.readFile(path.join(root, '404.html'));
        response.writeHead(404, { 'Content-Type': types['.html'], 'Cache-Control': 'no-store' }).end(content);
    }
});

server.on('error', error => {
    console.error('Could not start the preview server:', error.message);
    process.exitCode = 1;
});
server.listen(port, '127.0.0.1', () => console.log('Cloudvana preview: http://127.0.0.1:' + server.address().port));
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => server.close());
