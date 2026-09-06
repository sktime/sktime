/* Copy to clipboard functionality for parameter names (scikit-learn 1.7+ compatible) */
function copyToClipboard(text, element) {
  var toggleableContent = element.closest('.sk-toggleable__content');
  var paramPrefix = toggleableContent && toggleableContent.dataset.paramPrefix
    ? toggleableContent.dataset.paramPrefix : '';
  var fullParamName = paramPrefix ? paramPrefix + text : text;

  var originalStyle = element.style.cssText;
  var originalHTML = element.innerHTML ? element.innerHTML.replace('Copied!', '') : '';

  navigator.clipboard.writeText(fullParamName)
    .then(function() {
      element.style.color = 'green';
      element.innerHTML = 'Copied!';
      setTimeout(function() {
        element.innerHTML = originalHTML;
        element.style.cssText = originalStyle;
      }, 2000);
    })
    .catch(function(err) {
      console.error('Failed to copy:', err);
      element.style.color = 'red';
      element.innerHTML = 'Failed!';
      setTimeout(function() {
        element.innerHTML = originalHTML;
        element.style.cssText = originalStyle;
      }, 2000);
    });
  return false;
}
