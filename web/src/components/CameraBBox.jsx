import { Fragment, h } from 'preact';
import ActivityIndicator from './ActivityIndicator';
import { useApiHost } from '../api';
import useSWR from 'swr';
import { useCallback, useEffect, useMemo, useRef, useState } from 'preact/hooks';
import { useResizeObserver } from '../hooks';

export default function CameraBBox({ camera, onload, searchParams = '', stretch = false }) {
  const { data: config } = useSWR('config');
  const apiHost = useApiHost();
  const [hasLoaded, setHasLoaded] = useState(false);
  const [bBoxList, setBBoxList] = useState([]);
  const containerRef = useRef(null);

  const { name } = config ? config.cameras[camera] : '';
  const enabled = config ? config.cameras[camera].enabled : 'True';
  const { width, height } = config ? config.cameras[camera].detect : { width: 1, height: 1 };

  const fetchBBox = async () => {
    try {
      const response = await fetch(`${apiHost}api/${name}/latest_results?${searchParams ? `&${searchParams}` : ''}`);
      const bboxListResponse = await response.json();
      setBBoxList(bboxListResponse['objects'])
    } catch (error) {
      console.error('Error:', error);
    }
  }

  useEffect(() => {
    if (!config) {
      return;
    }
    fetchBBox();
  }, [apiHost, name, searchParams, config]);

  useEffect(() => {
    setHasLoaded(true);
    onload && onload();
  }, [bBoxList, setHasLoaded])

  return (
    <Fragment>
      <div className="absolute z-3 w-full h-full left-0 top-0 flex justify-center" ref={containerRef}>
        {enabled ? (
          <div className="absolute z-3 w-full h-full left-0 top-0">
            <svg width="100%" height="100%" viewBox={'0 0 ' + width + ' ' + height}>
              {
                bBoxList.map((item) => (
                  <rect x={item['box'][0]} y={item['box'][1]} width={item['box'][2] - item['box'][0]} height={item['box'][3] - item['box'][1]}
                    fill="none" stroke={`rgb(${item['color'][2]},${item['color'][1]},${item['color'][0]})`} stroke-width={item['thickness'] + 3} />
                ))
              }
            </svg>
          </div>
        ) : (
          <div class="text-center pt-6">Camera is disabled in config, no stream or snapshot available!</div>
        )}
        {!hasLoaded && enabled ? (
          <div className="absolute z-3 w-full h-full left-0 top-0 flex justify-center" ref={containerRef}>
            <ActivityIndicator />
          </div>
        ) : null}
      </div>
    </Fragment>
  );
}
