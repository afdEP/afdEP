// @ts-check
const path = require('path');

const repo = 'https://github.com/afdEP/afdEP-comm';

// Prism theme limited to the logo palette.
const prism = {
  plain: {color: '#FFFFFF', backgroundColor: '#000000'},
  styles: [
    {types: ['comment', 'prolog'], style: {color: '#9E9D9D', fontStyle: 'italic'}},
    {types: ['keyword', 'builtin', 'function', 'boolean', 'number'], style: {color: '#FCBA00'}},
    {types: ['string', 'char', 'attr-value'], style: {color: '#9E9D9D'}},
    {types: ['punctuation', 'operator'], style: {color: '#9E9D9D'}},
  ],
};

/** @type {import('@docusaurus/types').Config} */
module.exports = {
  title: 'afdEP-comm',
  tagline: 'Expert-parallel communication for Attention-FFN Disaggregation',
  url: 'https://afdep.github.io',
  // PR previews build under pr-preview/pr-N/ (.github/workflows/docs.yml).
  baseUrl: process.env.BASE_URL || '/afdEP-comm/',
  organizationName: 'afdEP',
  projectName: 'afdEP-comm',
  trailingSlash: false,
  onBrokenLinks: 'throw',
  staticDirectories: ['../docs/assets'],
  markdown: {
    format: 'detect',
    hooks: {
      // Links that leave docs/ (README, examples, ...) point at the repository.
      onBrokenMarkdownLinks: ({sourceFilePath, url}) =>
        `${repo}/blob/main/${path.posix.join('website', path.posix.dirname(sourceFilePath), url)}`,
    },
  },
  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {path: '../docs', routeBasePath: 'docs'},
        blog: false,
        theme: {customCss: './src/css/custom.css'},
      }),
    ],
  ],
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {defaultMode: 'dark', disableSwitch: true, respectPrefersColorScheme: false},
      navbar: {
        logo: {alt: 'afdEP-comm', src: 'afdep-logo.png'},
        items: [
          {type: 'docSidebar', sidebarId: 'defaultSidebar', label: 'Docs', position: 'left'},
          {href: repo, label: 'GitHub', position: 'right'},
        ],
      },
      prism: {theme: prism, darkTheme: prism, additionalLanguages: ['bash']},
    }),
};
