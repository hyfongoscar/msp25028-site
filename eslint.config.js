import { fixupConfigRules, fixupPluginRules } from '@eslint/compat';
import { FlatCompat } from '@eslint/eslintrc';
import eslint from '@eslint/js';
import { globalIgnores } from 'eslint/config';
import _import from 'eslint-plugin-import';
import prettier from 'eslint-plugin-prettier';
import react from 'eslint-plugin-react';
import simpleImportSort from 'eslint-plugin-simple-import-sort';
import fs from 'fs';
import path, { dirname } from 'path';
import tseslint from 'typescript-eslint';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
  recommendedConfig: eslint.configs.recommended,
  allConfig: eslint.configs.all,
});

const prettierOptions = JSON.parse(
  fs.readFileSync(path.resolve(__dirname, '.prettierrc'), 'utf8'),
);

export default tseslint.config([
  {
    extends: fixupConfigRules(
      compat.extends(
        'eslint:recommended',
        'plugin:react/recommended',
        'plugin:react-hooks/recommended',
        'prettier',
      ),
    ),
    plugins: {
      react: fixupPluginRules(react),
      import: fixupPluginRules(_import),
      prettier,
      'simple-import-sort': simpleImportSort,
    },
    rules: {
      'import/order': ['off'],
      'react/display-name': 'off',
      'react/jsx-sort-props': 'warn',
      'prettier/prettier': ['error', prettierOptions],

      'simple-import-sort/imports': [
        'warn',
        {
          groups: [
            ['^react$'],
            ['^\\u0000'],
            ['^@?\\w'],
            ['^components(/.*|$)'],
            ['^containers(/.*|$)'],
            ['^(types|utils|api|config|styles|pages)(/.*|$)'],
            ['^\\.'],
          ],
        },
      ],

      'simple-import-sort/exports': 'warn',
      'import/first': 'warn',
      'import/newline-after-import': 'warn',
      'import/no-duplicates': 'error',

      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',
    },
  },
  {
    extends: tseslint.configs.recommended,
    rules: {
      'no-unused-vars': 'off',
      '@typescript-eslint/no-unused-vars': [
        'warn',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_|React',
        },
      ],
      '@typescript-eslint/no-explicit-any': 'off',
    },
  },
  {
    files: ['**/*.ts?(x)'],
    rules: {
      'prettier/prettier': ['warn', prettierOptions],
    },
  },
  globalIgnores(['**/*.lib.js', '**/*.min.js']),
]);
